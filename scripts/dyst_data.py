import os
import numpy as np

from pathlib import Path
from typing import List, Union, Dict
from datetime import datetime
from dysts.base import get_attractor_list, make_trajectory_ensemble, init_cond_sampler
from dysts.utils import standardize_ts
from gluonts.dataset.arrow import ArrowWriter
import importlib


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
        # and time_series.ndim == 2
    )

    # GluonTS requires this datetime format for reading arrow file
    # TODO: we don't need to store start time, get rid of this?
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dataset = [
        {"start": start, "target": ts} for ts in time_series
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


def process_trajs(base_dir: str, timeseries: Dict[str, np.ndarray]) -> None:
    """
    Saves each trajectory in timeseries ensemble to a separate arrow file
    """
    for sys_name, trajectory in timeseries.items():
        path = os.path.join(base_dir, f"{sys_name}.arrow")
        convert_to_arrow(path, trajectory)
    

def make_single_dyst(
        dyst_name: str = "Lorenz", 
        split: str = "train",
        num_points: int = 1024,
        num_periods: int = 5,
        num_ics: int = 5,
        rseed: int = 999,
) -> None:
    """
    Makes single dyst trajectories using the dysts.base.make_trajectory_ensemble functionality
    A bit hacky, but may be useful for testing
    """
    # initial conditions sampler
    sampler = init_cond_sampler()
    
    # make trajectory ensembles by aggregating ensemble for num_ics initial condition sample instances
    single_ensemble_list = []

    for _ in range(num_ics):
        # each ensemble is of type Dict[str, [ndarray]]
        single_ensemble = make_trajectory_ensemble(
            num_points, subset=[dyst_name], init_conds=sampler(scale=1e-1), param_transform=None,
            use_tqdm=True, standardize=False, pts_per_period=num_points//num_periods, # random_state=rseed,
        )
        print(single_ensemble[dyst_name][:5])
        # add the ensemble dicts to respective list
        single_ensemble_list.append(single_ensemble)

    # Aggregate results (1 system x 20 ics iter loop for each train and test ensemble aggregation)
    single_ensemble = {dyst_name: np.stack([d[dyst_name] for d in single_ensemble_list], axis=0).squeeze()}
    # single_ensemble = {dyst_name: standardize_ts(np.stack([d[dyst_name] for d in single_ensemble_list], axis=0))}
    print(single_ensemble[dyst_name].shape)
    # set up save directory
    data_dir = os.path.join(WORK_DIR, "data", split)
    os.makedirs(data_dir, exist_ok=True)

    # save trajectories to arrow file
    process_trajs(data_dir, single_ensemble)


def main():

    rseed = 999
    num_periods = 5
    num_points = 1024
    num_ics = 20

    test, train = split_systems(0.3, seed=rseed)    
    print(train)
    print(test)

    # initial conditions sampler
    sampler = init_cond_sampler()

    # make trajectory ensembles by aggregating ensemble for num_ics initial condition sample instances
    train_ensemble_list = []
    test_ensemble_list = []

    for _ in range(num_ics):
        # each ensemble is of type Dict[str, [ndarray]]
        train_ensemble = make_trajectory_ensemble(
            num_points, subset=train, use_multiprocessing=True, init_conds=sampler(), param_transform=None,
            use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods, random_state=rseed,
        )
        test_ensemble = make_trajectory_ensemble(
            num_points, subset=test, use_multiprocessing=True, init_conds=sampler(), param_transform=None,
            use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods, random_state=rseed,
        )

        # add the ensemble dicts to respective list
        train_ensemble_list.append(train_ensemble)
        test_ensemble_list.append(test_ensemble)

    # Aggregate results (136 systems x 20 ics iter loop for each train and test ensemble aggregation)
    train_ensemble = {key: np.stack([d[key] for d in train_ensemble_list], axis=0) for key in train_ensemble_list[0]}
    test_ensemble = {key: np.stack([d[key] for d in test_ensemble_list], axis=0) for key in test_ensemble_list[0]}

    print("Saving timeseries to arrow files")
    for split, ensemble in [('train', train_ensemble), ('test', test_ensemble)]:
        # set up save directory
        data_dir = os.path.join(WORK_DIR, f'data/{split}')
        os.makedirs(data_dir, exist_ok=True)
        # save trajectories to arrow file
        process_trajs(data_dir, ensemble)


if __name__ == '__main__':
    # main()
    make_single_dyst(dyst_name="Lorenz", split="train", num_ics=1)


