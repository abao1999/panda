import os
import warnings
from typing import Dict, List, Optional

import numpy as np
from dysts.systems import make_trajectory_ensemble
from tqdm import trange

from dystformer.sampling import (
    GaussianParamSampler,
    InstabilityEvent,
    OnAttractorInitCondSampler,
    TimeLimitEvent,
)
from dystformer.utils import (
    filter_dict,
    process_trajs,
)


class DystData:
    """
    Class to generate and save trajectory ensembles for a given set of dynamical systems.
    """

    def __init__(
        self,
        rseed: int = 999,
        num_periods: int = 5,
        num_points: int = 1024,
        num_ics: int = 3,
        num_param_perturbations: int = 1,
        events: Optional[List] = None,
        verbose: bool = True,
    ):
        self.rseed = rseed
        self.num_periods = num_periods
        self.num_points = num_points
        self.num_ics = num_ics
        self.num_param_perturbations = num_param_perturbations
        self.events = events
        self.verbose = verbose

        if events is None:
            warnings.warn(
                "No events provided for numerical integration. Defaulting to TimeLimitEvent with max_duration of 5 minutes \
                and InstabilityEvent with threshold of 1e4."
            )
            # events for solve_ivp
            time_limit_event = TimeLimitEvent(max_duration=60 * 5)  # 5 min time limit
            instability_event = InstabilityEvent(threshold=1e4)
            self.events = [time_limit_event, instability_event]

        # NOTE: we are fixing the sampler settings here for now to ensure consistency across runs
        # Sampler to generate a perturbed parameter set for each dynamical system
        self.param_sampler = GaussianParamSampler(
            random_seed=rseed, scale=1e-1, verbose=verbose
        )
        # Sampler to generate initial conditions for each dynamical system
        self.ic_sampler = OnAttractorInitCondSampler(
            reference_traj_length=1024,
            reference_traj_transient=200,
            events=self.events,
            verbose=verbose,
        )

    def save_dyst_ensemble(
        self,
        dysts_names: List[str],
        split: str = "train",
        samples_save_interval: int = 1,
        save_dir: str = ".",
    ):
        print(
            f"Making {split} split with {len(dysts_names)} dynamical systems: \n {dysts_names}"
        )

        save_dyst_dir = os.path.join(save_dir, split)

        num_total_samples = self.num_param_perturbations * self.num_ics
        ensemble_list = []

        # TODO: for random parameter perturbations, need to check validity, as we currently get nans, which throws error at the dysts level
        for i in range(self.num_param_perturbations):
            for j in trange(self.num_ics):
                sample_idx = i * self.num_ics + j

                print("Making ensemble for sample ", sample_idx)
                # each ensemble is of type Dict[str, [ndarray]]
                ensemble = make_trajectory_ensemble(
                    self.num_points,
                    resample=True,
                    subset=dysts_names,
                    use_multiprocessing=True,
                    ic_transform=self.ic_sampler if self.num_ics > 1 else None,
                    param_transform=self.param_sampler
                    if self.num_param_perturbations > 1
                    else None,
                    use_tqdm=True,
                    standardize=False,
                    pts_per_period=self.num_points // self.num_periods,
                    events=self.events,
                    rng=self.param_sampler.rng,
                )

                ensemble, excluded_keys = filter_dict(ensemble)
                print("INTEGRATION FAILED FOR:", excluded_keys)

                ensemble_list.append(ensemble)

                # save samples of trajectory ensembles to arrow files and clear list of ensembles
                # Essentially a batched version of process_trajs
                if ((sample_idx + 1) % samples_save_interval) == 0 or (
                    sample_idx + 1
                ) == num_total_samples:
                    self._save_ensembles(
                        ensemble_list, save_dyst_dir, verbose=self.verbose
                    )
                    ensemble_list = []

    def _save_ensembles(
        self,
        ensemble_list: List[Dict[str, np.ndarray]],
        data_dir: str,
        verbose: bool = True,
    ):
        print(
            f"Saving {len(ensemble_list)} sampled trajectories to arrow files within {data_dir}"
        )
        # transpose and stack to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
        ensemble_keys = set().union(*[d.keys() for d in ensemble_list])
        ensemble = {
            key: np.stack(
                [d[key] for d in ensemble_list if key in d], axis=0
            ).transpose(0, 2, 1)
            for key in ensemble_keys
        }

        os.makedirs(data_dir, exist_ok=True)
        process_trajs(data_dir, ensemble, verbose=verbose)
