import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
from dysts.sampling import BaseSampler
from dysts.systems import make_trajectory_ensemble
from tqdm import trange

from dystformer.attractor import (
    EnsembleCallbackHandler,
    check_boundedness,
    check_no_nans,
    check_not_fixed_point,
    check_not_limit_cycle,
    check_power_spectrum,
    check_stationarity,
)
from dystformer.sampling import (
    InstabilityEvent,
    TimeLimitEvent,
)
from dystformer.utils import (
    filter_dict,
    process_trajs,
)


@dataclass
class DystData:
    """
    Class to generate and save trajectory ensembles for a given set of dynamical systems.
    Args:
        rseed: random seed for reproducibility
        num_periods: number of periods to generate for each system
        num_points: number of points to generate for each system
        num_ics: number of initial conditions to generate for each system
        num_param_perturbations: number of parameter perturbations to generate for each system
        split_coords: whether to split the coordinates into two dimensions
        events: events to pass to solve_ivp
        apply_attractor_tests: whether to apply attractor tests
        verbose: whether to print verbose output
        debug_mode: flag to save failed trajectory ensembles for debugging
    """

    rseed: int = 999
    num_periods: int = 5
    num_points: int = 1024

    param_sampler: Optional[BaseSampler] = None
    ic_sampler: Optional[BaseSampler] = None
    num_ics: int = 3
    num_param_perturbations: int = 1

    split_coords: bool = True  # by default save trajectories compatible with Chronos
    events: Optional[List] = None
    apply_attractor_tests: bool = False
    verbose: bool = True
    debug_mode: bool = False

    def __post_init__(self):
        if self.events is None:
            warnings.warn(
                "No events provided for numerical integration. Defaulting to TimeLimitEvent with max_duration of 5 minutes \
                and InstabilityEvent with threshold of 1e4."
            )
            # events for solve_ivp
            time_limit_event = TimeLimitEvent(max_duration=60 * 5)  # 5 min time limit
            instability_event = InstabilityEvent(threshold=1e4)
            self.events = [time_limit_event, instability_event]

        # ensure that we can still generate data even if no samplers are provided
        if self.param_sampler is None:
            self.num_param_perturbations = 1
        if self.ic_sampler is None:
            self.num_ics = 1

        # Callbacks to check attractor properties
        if self.apply_attractor_tests:
            self.attractor_validator = self._build_attractor_validator()
        else:
            self.attractor_validator = None
            if self.num_param_perturbations > 1:
                warnings.warn(
                    "No attractor tests specified. Parameter perturbations may not result in valid attractors!"
                )

        # track the failed numerical integrations
        self.failed_integrations = defaultdict(list)

    def _build_attractor_validator(self) -> EnsembleCallbackHandler:
        """
        Builds a list of attractor tests to check for each trajectory ensemble.
        """
        print("Setting up callbacks to test attractor properties")
        # callbacks to check attractor validity when creating traj ensemble of dysts
        ens_callback_handler = EnsembleCallbackHandler(verbose=1)  # verbose=2
        ens_callback_handler.add_callback(check_no_nans)
        ens_callback_handler.add_callback(check_boundedness)
        ens_callback_handler.add_callback(check_not_fixed_point)
        ens_callback_handler.add_callback(
            partial(
                check_not_limit_cycle,
                tolerance=1e-3,
                min_recurrences=5,
            )
        )
        ens_callback_handler.add_callback(
            partial(
                check_power_spectrum,
                plot_save_dir=None,  # FIGS_SAVE_DIR # NOTE: set to None when actually generating data so we don't plot thousands of times
            )
        )
        ens_callback_handler.add_callback(
            partial(
                check_stationarity,
                method="recurrence",  # "statsmodels", # adfuller and kpss only maybe reliable for long horizon
            )
        )

        return ens_callback_handler

    def save_dyst_ensemble(
        self,
        dysts_names: List[str],
        split: str = "train",
        samples_save_interval: int = 1,
        save_dir: str = ".",
    ) -> None:
        """
        Save a trajectory ensemble for each sample instance (i.e. perturbation and initial condition).
        If enabled, attractor tests are performed on the ensembles to check if they are valid attractors.
        Trajectories are saved to Arrow files in the specified save directory.
        If the attractor tests are enabled, failed trajectories are also saved to Arrow files in a 'failed_attractors' subdirectory.
        Args:
            dysts_names: list of dyst names
            split: split to save the ensembles
            samples_save_interval: interval to save the ensembles
            save_dir: directory to save the ensembles
        """
        print(
            f"Making {split} split with {len(dysts_names)} dynamical systems: \n {dysts_names}"
        )

        save_dyst_dir = os.path.join(save_dir, split)
        os.makedirs(save_dyst_dir, exist_ok=True)
        if self.debug_mode:
            failed_dyst_dir = os.path.join(save_dir, "failed_attractors")
            os.makedirs(failed_dyst_dir, exist_ok=True)

        num_total_samples = self.num_param_perturbations * self.num_ics
        ensemble_list = []

        # TODO: for random parameter perturbations, need to check validity, as we currently get nans, which throws error at the dysts level
        # make a stream of rngs for each parameter perturbation
        pp_rng_stream = np.random.default_rng(self.rseed).spawn(
            self.num_param_perturbations,
        )
        for i, param_rng in zip(range(self.num_param_perturbations), pp_rng_stream):
            if self.param_sampler is not None:
                self.param_sampler.set_rng(param_rng)

            for j in trange(self.num_ics):
                sample_idx = i * self.num_ics + j

                # reset events that have a reset method
                if self.events is not None:
                    for event in self.events:
                        if hasattr(event, "reset"):
                            print("Resetting event: ", event)
                            event.reset()

                print("Making ensemble for sample ", sample_idx)
                # each ensemble is of type Dict[str, [ndarray]]
                ensemble = make_trajectory_ensemble(
                    self.num_points,
                    resample=True,
                    subset=dysts_names,
                    use_multiprocessing=True,
                    ic_transform=self.ic_sampler if sample_idx > 0 else None,
                    param_transform=self.param_sampler if i > 0 else None,
                    ic_rng=param_rng,
                    use_tqdm=True,
                    standardize=False,
                    pts_per_period=self.num_points // self.num_periods,
                    events=self.events,
                )

                ensemble, excluded_keys = filter_dict(ensemble)
                if len(excluded_keys) > 0:
                    warnings.warn(f"INTEGRATION FAILED FOR: {excluded_keys}")
                    for dyst_name in excluded_keys:
                        # NOTE: to get more fine-grained information about the failed event, we need to modify dysts level
                        self.failed_integrations[dyst_name].append(sample_idx)

                successful_dyst_names = list(ensemble.keys())
                print(f"Successful Dysts: {successful_dyst_names}")
                if len(successful_dyst_names) == 0:
                    print(
                        "No successful trajectories for this sample. Skipping, will not save to arrow files."
                    )
                    continue

                ensemble_list.append(ensemble)

                # Batched version of process_trajs: save sampled ensembles to arrow files and clear list of ensembles
                is_last_sample = (sample_idx + 1) == num_total_samples
                if ((sample_idx + 1) % samples_save_interval) == 0 or is_last_sample:
                    # process the ensemble list
                    ensemble, failed_ensemble = self._process_ensemble_list(
                        ensemble_list, sample_idx
                    )
                    # save the processed ensemble to arrow files
                    print(
                        f"Saving all valid sampled trajectories from {len(ensemble)} systems to arrow files within {save_dyst_dir}"
                    )
                    process_trajs(
                        save_dyst_dir,
                        ensemble,
                        split_coords=self.split_coords,
                        verbose=self.verbose,
                    )

                    # also save the failed ensembles to arrow files in a failures subdirectory
                    if failed_ensemble is not None and self.debug_mode:
                        print(
                            f"Saving all failed sampled trajectories from {len(failed_ensemble)} systems to arrow files within {failed_dyst_dir}"
                        )
                        process_trajs(
                            failed_dyst_dir,
                            failed_ensemble,
                            split_coords=self.split_coords,
                            verbose=self.verbose,
                        )
                    # clear the ensemble list to reset
                    ensemble_list = []

    def _process_ensemble_list(
        self,
        ensemble_list: List[Dict[str, np.ndarray]],
        sample_idx: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Process the ensemble list by checking for valid attractors and filtering out invalid ones.
        Also, transposes and stacks trajectories to get shape (num_samples, num_dims, num_timesteps).
        """
        print(f"Processing {len(ensemble_list)} ensembles")
        # transpose and stack to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
        ensemble = {
            key: np.stack(
                [d[key] for d in ensemble_list if key in d], axis=0
            ).transpose(0, 2, 1)
            for key in [key for d in ensemble_list for key in d.keys()]
        }
        failed_ensemble = {}

        print(f"Checking if attractor properties are valid for {len(ensemble)} systems")
        # Check if attractor properties are valid
        if self.attractor_validator is not None:
            # Filter out invalid attractors and add valid attractors to ensemble list
            ensemble, failed_ensemble = self.attractor_validator.filter_ensemble(
                ensemble, first_sample_idx=sample_idx
            )
        return ensemble, failed_ensemble

    def save_summary(self, save_json_path: str):
        """
        Save a summary of valid attractor counts and failed checks to a json file.
        """
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        print(f"Saving summary to {save_json_path}")

        if self.attractor_validator is None:
            summary_dict = {"failed_integrations": self.failed_integrations}

        else:
            valid_dyst_counts = self.attractor_validator.valid_dyst_counts
            failed_checks = self.attractor_validator.failed_checks
            summary_dict = {
                "valid_dyst_counts": valid_dyst_counts,
                "failed_checks": failed_checks,
                "failed_integrations": self.failed_integrations,
            }

        with open(save_json_path, "w") as f:
            # Add the attractor validator summary to the json file
            json.dump(
                summary_dict,
                f,
                indent=4,
            )  # Added indent for pretty printing
            f.write("\n")
