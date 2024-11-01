import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dysts.sampling import BaseSampler
from dysts.systems import make_trajectory_ensemble
from tqdm import trange

from dystformer.attractor import (
    AttractorValidator,
    check_boundedness,
    check_lyapunov_exponent,
    check_not_fixed_point,
    check_not_limit_cycle,
    check_not_trajectory_decay,
    check_power_spectrum,
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
        attractor_validator_kwargs: kwargs for the attractor validator
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
    attractor_validator_kwargs: Dict[str, Any] = field(default_factory=dict)
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

        # attractor tests
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

    def _build_attractor_validator(self) -> AttractorValidator:
        """
        Builds a list of attractor tests to check for each trajectory ensemble.
        """
        print("Setting up callbacks to test attractor properties")
        validator = AttractorValidator(**self.attractor_validator_kwargs)
        validator.add_test_fn(
            partial(check_boundedness, threshold=1e3, max_num_stds=10)
        )
        validator.add_test_fn(partial(check_not_fixed_point, atol=1e-3, tail_prop=0.1))
        validator.add_test_fn(
            partial(check_not_trajectory_decay, atol=1e-3, tail_prop=0.5)
        )
        # for STRICT MODE (strict criteria for detecting limit cycles), try:
        # min_prop_recurrences = 0.1, min_counts_per_rtime = 100, min_block_length=50, min_recurrence_time = 10, enforce_endpoint_recurrence = True,
        validator.add_test_fn(
            partial(
                check_not_limit_cycle,
                tolerance=1e-3,
                min_counts_per_rtime=100,
                min_block_length=20,
            )
        )
        validator.add_test_fn(
            partial(
                check_power_spectrum,
                rel_peak_height_threshold=1e-5,
                rel_prominence_threshold=None,
            )
        )
        validator.add_test_fn(check_lyapunov_exponent)
        return validator

    def save_dyst_ensemble(
        self,
        dysts_names: List[str],
        split: str = "train",
        split_failures: str = "failed_attractors",
        samples_process_interval: int = 1,
        save_dir: Optional[str] = None,
        standardize: bool = False,
        use_multiprocessing: bool = True,
        reset_attractor_validator: bool = False,
        **kwargs,
    ) -> None:
        print(
            f"Making {split} split with {len(dysts_names)} dynamical systems: \n {dysts_names}"
        )

        if self.attractor_validator is not None and reset_attractor_validator:
            self.attractor_validator.reset()
            self.failed_integrations.clear()

        save_dyst_dir, failed_dyst_dir = self._prepare_save_directories(
            save_dir, split, split_failures=split_failures
        )
        num_total_samples = self.num_param_perturbations * self.num_ics

        def process_and_save_callback():
            """
            Callback to process and save ensembles.
            """
            ensemble_list = []

            def _callback(ensemble, excluded_keys, sample_idx):
                if len(ensemble.keys()) == 0:
                    print(
                        "No successful trajectories for this sample. Skipping, will not save to arrow files."
                    )
                    return

                ensemble_list.append(ensemble)

                is_last_sample = (sample_idx + 1) == num_total_samples
                if ((sample_idx + 1) % samples_process_interval) == 0 or is_last_sample:
                    self._process_and_save_ensemble(
                        ensemble_list, sample_idx, save_dyst_dir, failed_dyst_dir
                    )
                    ensemble_list.clear()

            return _callback

        def handle_failed_integrations_callback(ensemble, excluded_keys, sample_idx):
            """
            Callback to handle failed integrations.
            """
            if len(excluded_keys) > 0:
                warnings.warn(f"INTEGRATION FAILED FOR: {excluded_keys}")
                for dyst_name in excluded_keys:
                    self.failed_integrations[dyst_name].append(sample_idx)

        _ = self._generate_ensembles(
            dysts_names,
            [handle_failed_integrations_callback, process_and_save_callback()],
            standardize=standardize,
            use_multiprocessing=use_multiprocessing,
            **kwargs,
        )

    def _generate_ensembles(
        self,
        dysts_names: List[str],
        postprocessing_callbacks: List[Callable] = [],
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        ensembles = []
        pp_rng_stream = np.random.default_rng(self.rseed).spawn(
            self.num_param_perturbations
        )

        for i, param_rng in zip(range(self.num_param_perturbations), pp_rng_stream):
            if self.param_sampler is not None:
                self.param_sampler.set_rng(param_rng)

            if self.ic_sampler is not None:
                self.ic_sampler.clear_cache()

            for j in trange(self.num_ics):
                sample_idx = i * self.num_ics + j

                # reset events that have a reset method
                self._reset_events()

                print("Making ensemble for sample ", sample_idx)
                ensemble = make_trajectory_ensemble(
                    self.num_points,
                    subset=dysts_names,
                    ic_transform=self.ic_sampler if sample_idx > 0 else None,
                    param_transform=self.param_sampler if i > 0 else None,
                    ic_rng=param_rng,
                    use_tqdm=True,
                    pts_per_period=self.num_points // self.num_periods,
                    events=self.events,
                    **kwargs,
                )
                ensemble, excluded_keys = filter_dict(ensemble)
                ensembles.append(ensemble)

                for callback in postprocessing_callbacks:
                    callback(ensemble, excluded_keys, sample_idx)

        return ensembles

    def _reset_events(self) -> None:
        if self.events is not None:
            for event in self.events:
                if hasattr(event, "reset") and callable(event.reset):
                    print("Resetting event: ", event)
                    event.reset()

    def _prepare_save_directories(
        self,
        save_dir: Optional[str],
        split: str,
        split_failures: str = "failed_attractors",
    ) -> Tuple[Optional[str], Optional[str]]:
        if save_dir is not None:
            save_dyst_dir = os.path.join(save_dir, split)
            os.makedirs(save_dyst_dir, exist_ok=True)
            if self.debug_mode:
                failed_dyst_dir = os.path.join(save_dir, split_failures)
                os.makedirs(failed_dyst_dir, exist_ok=True)
                print(f"valid attractors will be saved to {save_dyst_dir}")
                print(f"failed attractors will be saved to {failed_dyst_dir}")
            else:
                failed_dyst_dir = None
        else:
            warnings.warn("save_dir is None, will not save trajectories.")
            save_dyst_dir = failed_dyst_dir = None
        return save_dyst_dir, failed_dyst_dir

    def _process_and_save_ensemble(
        self,
        ensemble_list: List[Dict[str, np.ndarray]],
        sample_idx: int,
        save_dyst_dir: Optional[str],
        failed_dyst_dir: Optional[str],
    ) -> None:
        ensemble, failed_ensemble = self._process_ensemble_list(
            ensemble_list, sample_idx
        )

        if save_dyst_dir is not None:
            print(
                f"Saving all valid sampled trajectories from {len(ensemble)} systems to arrow files within {save_dyst_dir}"
            )
            process_trajs(
                save_dyst_dir,
                ensemble,
                split_coords=self.split_coords,
                verbose=self.verbose,
            )
            if failed_ensemble and failed_dyst_dir is not None:
                print(
                    f"Saving all failed sampled trajectories from {len(failed_ensemble)} systems to arrow files within {failed_dyst_dir}"
                )
                process_trajs(
                    failed_dyst_dir,
                    failed_ensemble,
                    split_coords=self.split_coords,
                    verbose=self.verbose,
                )

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
        if self.attractor_validator is not None:
            ensemble, failed_ensemble = (
                self.attractor_validator.multiprocessed_filter_ensemble(
                    ensemble, first_sample_idx=sample_idx
                )
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
            failed_samples = self.attractor_validator.failed_samples
            valid_samples = self.attractor_validator.valid_samples
            summary_dict = {
                "num_parameter_successes": sum(
                    np.unique(np.array(sample_inds).astype(int) // self.num_ics).shape[
                        0
                    ]
                    for sample_inds in valid_samples.values()
                ),
                "num_total_candidates": self.num_param_perturbations
                * len(
                    valid_samples.keys()
                    | failed_samples.keys()
                    | self.failed_integrations.keys()
                ),
                "valid_dyst_counts": valid_dyst_counts,
                "failed_checks": failed_checks,
                "failed_integrations": self.failed_integrations,
                "failed_samples": failed_samples,
                "valid_samples": valid_samples,
            }

        with open(save_json_path, "w") as f:
            json.dump(summary_dict, f, indent=4)
            f.write("\n")
