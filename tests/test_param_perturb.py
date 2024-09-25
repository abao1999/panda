import os
import numpy as np
from typing import List, Optional, Callable
from tqdm import trange
import argparse

from dysts.systems import make_trajectory_ensemble

from dystformer.sampling import (
    InstabilityEvent, 
    TimeLimitEvent,
    GaussianParamSampler,
    OnAttractorInitCondSampler,
)

from dystformer.utils import (
    filter_dict,
    process_trajs,
)


WORK_DIR = os.getenv('WORK', '')
DATA_DIR = os.path.join(WORK_DIR, 'data')


def save_dyst_ensemble(
    dysts_names: List[str] = ['Lorenz'],
    rseed: int = 999,
    num_periods: int = 5,
    num_points: int = 1024,
    num_ics: int = 3,
    num_param_perturbations: int = 1,
    samples_save_interval: int = 1,
    events: Optional[List[Callable]] = None,
    split: str = 'train',
) -> None:
    print(f"Making {len(dysts_names)} dynamical systems: \n {dysts_names}")
    
    param_sampler = GaussianParamSampler(random_seed=rseed, scale=1e-1, verbose=True)
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=1024,
        reference_traj_transient=200,
        random_seed=rseed, 
        events=events, 
        verbose=True
    )

    num_total_samples = num_param_perturbations * num_ics
    ensemble_list = []

    for i in range(num_param_perturbations):
        for j in trange(num_ics):
            sample_idx = i + j

            print("Making ensemble for sample ", sample_idx)
            # each ensemble is of type Dict[str, ndarray]
            ensemble = make_trajectory_ensemble(
                num_points, 
                resample=True,
                subset=dysts_names, 
                use_multiprocessing=True, 
                ic_transform=ic_sampler if num_ics > 1 else None,
                param_transform=param_sampler if num_param_perturbations > 1 else None,
                use_tqdm=True, 
                standardize=True, 
                pts_per_period=num_points//num_periods,
                events=events,
                rng=param_sampler.rng,
            )

            ensemble, excluded_keys = filter_dict(ensemble) #, req_num_vals=num_points)
            if excluded_keys:
                print("INTEGRATION FAILED FOR:", excluded_keys)

            ensemble_list.append(ensemble)

            # save samples of trajectory ensembles to arrow files and clear list of ensembles
            # Essentially a batched version of process_trajs
            if ((sample_idx + 1) % samples_save_interval) == 0 or (sample_idx + 1) == num_total_samples:
                # transpose and stack to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
                ensemble_keys = set().union(*[d.keys() for d in ensemble_list])
                ensemble = {key: np.stack([d[key] for d in ensemble_list if key in d], axis=0).transpose(0, 2, 1) for key in ensemble_keys}

                # save trajectories to arrow files
                data_dir = os.path.join(DATA_DIR, split)
                os.makedirs(data_dir, exist_ok=True)
                process_trajs(data_dir, ensemble, verbose=True)

                ensemble_list = []


if __name__ == '__main__':

    # For testing select systems
    parser = argparse.ArgumentParser()
    parser.add_argument("dysts_names", help="Names of the dynamical systems", nargs="+", type=str)
    args = parser.parse_args()

    dysts_names = args.dysts_names
    print("dysts_names: ", dysts_names)

    # set random seed
    rseed = 999 # we are using same seed for split and ic and param samplers

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=60*10)  # 1 min time limit
    instability_event = InstabilityEvent(threshold=1e3)

    # make the train split
    save_dyst_ensemble(
        dysts_names,
        rseed=rseed,
        num_periods=5,
        num_points=1024,
        num_ics=2,
        num_param_perturbations=2,
        events=[time_limit_event, instability_event],
        split='train',
    )