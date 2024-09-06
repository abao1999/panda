import os
import numpy as np
from dataclasses import dataclass
from tqdm import trange
from typing import List, Optional

from dysts.base import make_trajectory_ensemble, init_cond_sampler
from chronos_dysts.utils import (
    split_systems,
    process_trajs,
    filter_dict,
)


WORK_DIR = os.getenv('WORK')
DELAY_SYSTEMS = ['MackeyGlass', 'IkedaDelay', 'SprottDelay', 'VossDelay', 'ScrollDelay', 'PiecewiseCircuit']
# FORCED_SYSTEMS = ['ForcedFitzHughNagumo', 'ForcedBrusselator', 'ForcedVanDerPol']


@dataclass
class ParamPerturb:

    scale: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, name: str, param: np.ndarray) -> np.ndarray:
        size = None if np.isscalar(param) else param.shape
        return param + self.rng.normal(scale=self.scale*np.linalg.norm(param), size=size)

# Event function to check if integration is taking too long
import time
class TimeLimitEvent:
    def __init__(self, max_duration):
        self.start_time = None
        self.max_duration = max_duration

    def __call__(self, t, y):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration:
            print("Integration stopped due to time limit.")
            return 0  # Trigger the event
        return 1  # Continue the integration

# Event function to detect instability
def instability_event(t, y):
    # Example criterion: If the solution's magnitude exceeds a large threshold
    if np.any(np.abs(y) > 1e3): # reasonable threshold, since we are standardizing trajectories
        print("y: ", y)
        print("Integration stopped due to instability.")
        return 0  # Trigger the event
    return 1  # Continue the integration


def save_dyst_ensemble(
    dysts_names: List[str] = ['Lorenz'],
    split: str = "train",
    rseed: int = 999,
    num_periods: int = 5,
    num_points: int = 1024,
    num_ics: int = 3,
    num_param_perturbations: int = 1,
    samples_save_interval: int = 1,
    events: Optional[List] = None,
):
    print(f"Making {split} split with {len(dysts_names)} dynamical systems: \n {dysts_names}")
    
    ic_sampler = init_cond_sampler(subset=dysts_names, random_seed=rseed)
    param_sampler = ParamPerturb(scale=1e-3, random_seed=rseed)

    num_total_samples = num_param_perturbations * num_ics
    ensemble_list = []

    # TODO: for random parameter perturbations, need to check validity, as we currently get nans, which throws error at the dysts level
    for i in range(num_param_perturbations):
        for j in trange(num_ics):
            sample_idx = i + j

            print("Making ensemble for sample ", sample_idx)
            # each ensemble is of type Dict[str, [ndarray]]
            ensemble = make_trajectory_ensemble(
                num_points, subset=dysts_names, use_multiprocessing=True, 
                init_conds=ic_sampler(scale=1e-1), param_transform=param_sampler if num_param_perturbations > 1 else None,
                use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods,
                events=events,
            )
            ensemble, excluded_keys = filter_dict(ensemble) #, req_num_vals=num_points)
            print("INTEGRATION FAILED FOR:", excluded_keys)

            ensemble_list.append(ensemble)

            # save samples of trajectory ensembles to arrow files and clear list of ensembles
            if ((sample_idx + 1) % samples_save_interval) == 0 or (sample_idx + 1) == num_total_samples:
                print(f"Saving {len(ensemble_list)} sampled trajectoies to arrow files")
                # transpose and stack to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
                ensemble_keys = set().union(*[d.keys() for d in ensemble_list])
                ensemble = {key: np.stack([d[key] for d in ensemble_list if key in d], axis=0).transpose(0, 2, 1) for key in ensemble_keys}
                
                data_dir = os.path.join(WORK_DIR, f'data/{split}')
                os.makedirs(data_dir, exist_ok=True)
                process_trajs(data_dir, ensemble, verbose=True)
                # reset lists of ensembles
                ensemble_list = []

if __name__ == '__main__':

    # set random seed
    rseed = 999 # we are using same seed for split and ic and param samplers

    # generate split of dynamical systems
    test, train = split_systems(0.3, seed=rseed, excluded_systems=DELAY_SYSTEMS) # + FORCED_SYSTEMS)

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=120)  # 2 min time limit
    time_limit_event.terminal = True
    instability_event.terminal = True

    # make the train split
    save_dyst_ensemble(
        train, 
        split="train", 
        rseed=rseed,
        num_periods=5,
        num_points=1024,
        num_ics=5,
        num_param_perturbations=1,
        events=[time_limit_event, instability_event],
    )

    # make the test split
    save_dyst_ensemble(
        test, 
        split="test", 
        rseed=rseed,
        num_periods=5,
        num_points=1024,
        num_ics=5,
        num_param_perturbations=1,
        events=[time_limit_event, instability_event],
    )
