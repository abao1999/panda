import os
import numpy as np

from pathlib import Path
from typing import List, Union, Dict
from datetime import datetime
from dysts.base import get_attractor_list, make_trajectory_ensemble
from gluonts.dataset.arrow import ArrowWriter
import importlib


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
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
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
    for sys_name, trajectory in timeseries.items():
        path = os.path.join(base_dir, f"{sys_name}.arrow")
        convert_to_arrow(path, trajectory)
    

def main():

    rseed = 999
    num_periods = 5
    num_points = 1024

    test, train = split_systems(0.3, seed=rseed)    
    print(train)
    print(test)

    train_ensemble = make_trajectory_ensemble(
        num_points, subset=train, random_state=rseed, use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods,
    )
    test_ensemble = make_trajectory_ensemble(
        num_points, subset=test, random_state=rseed, use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods,
    )

    print("Saving timeseries to arrow files")    
    work_dir = os.getenv('WORK')

    for split, trajs in [('train', train_ensemble), ('test', test_ensemble)]:
        
        data_dir = os.path.join(work_dir, f'data/{split}')
        os.makedirs(data_dir, exist_ok=True)
        process_trajs(data_dir, trajs)


# TODO: once we add initial condition option for dysts.make_trajectory_ensemble, we need to modify this
def make_single_dyst(dyst_name="Lorenz", split="train"):
    """
    A test function to make a single [dyst_name].arrow file in data/train split
    """
    work_dir = os.getenv('WORK')
    data_dir = os.path.join(work_dir, 'data', split)
    os.makedirs(data_dir, exist_ok=True)
    num_points = 1024
    num_periods = 5
    dyst_module = importlib.import_module("dysts.flows")
    dyst_class_ = getattr(dyst_module, dyst_name)
    print(dyst_class_)
    traj = dyst_class_().make_trajectory(num_points, standardize=True, pts_per_period=num_points//num_periods)
    convert_to_arrow(os.path.join(data_dir, f"{dyst_name}.arrow"), traj)

if __name__ == '__main__':
    main()
    # make_single_dyst(dyst_name="Lorenz", split="train")


