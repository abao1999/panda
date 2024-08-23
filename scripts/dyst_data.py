import os
import numpy as np

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from dysts.base import get_attractor_list, make_trajectory_ensemble, init_cond_sampler
from gluonts.dataset.arrow import ArrowWriter
from typing import List, Union, Dict, Optional

from tqdm import trange

WORK_DIR = os.getenv('WORK')


def split_systems(prop: float, seed: int):
    systems = get_attractor_list()
    np.random.default_rng(seed).shuffle(systems)
    split = int(len(systems)*prop)
    return systems[:split], systems[split:]


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) 
        and time_series.ndim == 2
    )

    # GluonTS requires this datetime format for reading arrow file
    start = datetime.now().strftime("%Y-%m-%d %H:%M")

    dataset = [
        {"start": start, "target": ts} for ts in time_series
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


def process_trajs(base_dir: str, timeseries: Dict[str, np.ndarray], verbose: Optional[bool] = False) -> None:
    """Saves each trajectory in timeseries ensemble to a separate directory
    """
    for sys_name, trajectories in timeseries.items():
        if verbose: print(trajectories.shape)
        system_folder = os.path.join(base_dir, sys_name)
        os.makedirs(system_folder, exist_ok=True)

        # get the last sample index from the directory, so we can continue saving samples filenames with the correct index
        max_existing_sample_idx = -1
        for filename in os.listdir(system_folder):
            if filename.endswith(".arrow"):
                sample_idx = int(filename.split("_")[0])
                max_existing_sample_idx = max(max_existing_sample_idx, sample_idx)

        for i, trajectory in enumerate(trajectories): 
            curr_sample_idx = max_existing_sample_idx + i + 1
            
            if trajectory.ndim == 1: # handles case where just a single trajectory sample was generated and saved to timeseries dict
                trajectory = np.expand_dims(trajectory, axis=0) # trajectories.reshape(1, -1)
            if verbose:
                print(f"Saving {sys_name} trajectory {curr_sample_idx} with shape {trajectory.shape}")
            
            path = os.path.join(system_folder, f"{curr_sample_idx}_T-{trajectory.shape[-1]}.arrow")
            convert_to_arrow(path, trajectory)


@dataclass
class ParamPerturb:

    scale: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, name: str, param: np.ndarray) -> np.ndarray:
        size = None if np.isscalar(param) else param.shape
        return param + self.rng.normal(scale=self.scale*np.linalg.norm(param), size=size)


def main():

    rseed = 999
    num_periods = 5
    num_points = 1024
    num_ics = 4
    num_param_perturbations = 1

    test, train = split_systems(0.3, seed=rseed)   
    print(train)
    print(test)

    train_ic_sampler = init_cond_sampler(subset=train, random_seed=rseed)
    test_ic_sampler = init_cond_sampler(subset=test, random_seed=rseed)
    param_sampler = ParamPerturb(scale=1e-3, random_seed=rseed)

    num_total_samples = num_param_perturbations * num_ics
    samples_save_interal = 2
    train_ensemble_list = []
    test_ensemble_list = []

    # TODO: for random parameter perturbations, need to check validity, as we currently get nans, which throws error at the dysts level
    for i in range(num_param_perturbations):
        for j in trange(num_ics):
            sample_idx = i + j
            # each ensemble is of type Dict[str, [ndarray]]
            train_ensemble = make_trajectory_ensemble(
                num_points, subset=train, use_multiprocessing=True, 
                init_conds=train_ic_sampler(scale=1e-1), param_transform=param_sampler if num_param_perturbations > 1 else None,
                use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods,
            )
            test_ensemble = make_trajectory_ensemble(
                num_points, subset=test, use_multiprocessing=True, 
                init_conds=test_ic_sampler(scale=1e-1), param_transform=param_sampler if num_param_perturbations > 1 else None,
                use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods,
            )

            train_ensemble_list.append(train_ensemble)
            test_ensemble_list.append(test_ensemble)

            if ((sample_idx + 1) % samples_save_interal) == 0 or (sample_idx + 1) == num_total_samples: # save and clear list of ensembles
                assert len(train_ensemble_list) == len(test_ensemble_list), "Train and test ensemble lists should have same length"
                # transpose and stack to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
                train_ensemble = {key: np.stack([d[key].T for d in train_ensemble_list], axis=0) for key in train_ensemble_list[0]}
                test_ensemble = {key: np.stack([d[key].T for d in test_ensemble_list], axis=0) for key in test_ensemble_list[0]}
                print(f"Saving {len(train_ensemble_list)} sampled train and test trajectories to arrow files")
                for split, ensemble in [('train', train_ensemble), ('test', test_ensemble)]:
                    data_dir = os.path.join(WORK_DIR, f'data/{split}')
                    os.makedirs(data_dir, exist_ok=True)
                    process_trajs(data_dir, ensemble, verbose=True)
                # reset lists of ensembles
                train_ensemble_list = []
                test_ensemble_list = []

if __name__ == '__main__':
    main()

