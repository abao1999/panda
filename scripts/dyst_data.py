import os
import numpy as np

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from dysts.base import get_attractor_list, make_trajectory_ensemble, init_cond_sampler
from gluonts.dataset.arrow import ArrowWriter
from typing import List, Union, Dict, Iterable

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


def process_trajs(base_dir: str, timeseries: Dict[str, np.ndarray]) -> None:
    """Saves each trajectory in timeseries ensemble to a separate directory
    """
    for sys_name, trajectories in timeseries.items():
        system_folder = os.path.join(base_dir, sys_name)
        os.makedirs(system_folder, exist_ok=True)
        for i, trajectory in enumerate(trajectories): 
            path = os.path.join(system_folder, f"{i}_T-{trajectory.shape[-1]}.arrow")
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
    num_ics = 2
    num_param_perturbations = 2

    test, train = split_systems(0.3, seed=rseed)    
    test = train = ["Lorenz"]
    print(train)
    print(test)

    train_ic_sampler = init_cond_sampler(subset=train, random_seed=rseed)
    test_ic_sampler = init_cond_sampler(subset=train, random_seed=rseed)
    param_sampler = ParamPerturb(scale=1e-3, random_seed=rseed)

    # make trajectory ensembles by aggregating ensemble for num_ics initial condition sample instances
    train_ensemble_list = []
    test_ensemble_list = []

    for _ in range(num_ics):
        for _ in range(num_param_perturbations):
            # each ensemble is of type Dict[str, [ndarray]]
            train_ensemble = make_trajectory_ensemble(
                num_points, subset=train, use_multiprocessing=True, 
                init_conds=train_ic_sampler(scale=1e-2), param_transform=param_sampler,
                use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods, random_state=rseed,
            )
            test_ensemble = make_trajectory_ensemble(
                num_points, subset=test, use_multiprocessing=True, 
                init_conds=test_ic_sampler(scale=1e-2), param_transform=param_sampler,
                use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods, random_state=rseed,
            )

            train_ensemble_list.append(train_ensemble)
            test_ensemble_list.append(test_ensemble)

    train_ensemble = {key: np.stack([d[key].T for d in train_ensemble_list], axis=0) for key in train_ensemble_list[0]}
    test_ensemble = {key: np.stack([d[key].T for d in test_ensemble_list], axis=0) for key in test_ensemble_list[0]}

    print("Saving timeseries to arrow files")
    for split, ensemble in [('train', train_ensemble), ('test', test_ensemble)]:
        data_dir = os.path.join(WORK_DIR, f'data/{split}')
        os.makedirs(data_dir, exist_ok=True)
        process_trajs(data_dir, ensemble)


if __name__ == '__main__':
    main()

