import os
import dysts
import numpy as np
import dysts.flows as flows

from pathlib import Path
from typing import List, Union, Dict
from datetime import datetime
from dysts.base import get_attractor_list, make_trajectory_ensemble
from gluonts.dataset.arrow import ArrowWriter


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

    # Set an arbitrary start time
    dt = datetime.now().strftime("%m_%d_%H_%M")

    dataset = [
        {"start": dt, "target": ts} for ts in time_series
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

    train, test = split_systems(0.3, seed=rseed)    
    print(train)
    print(test)

    train_ensemble = make_trajectory_ensemble(
        num_points, subset=train, random_state=rseed, use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods
    )
    test_ensemble = make_trajectory_ensemble(
        num_points, subset=test, random_state=rseed, use_tqdm=True, standardize=True, pts_per_period=num_points//num_periods
    )

    print("Saving timeseries to arrow files")    
    work_dir = os.getenv('WORK')
    data_dir = os.path.join(work_dir, 'data')

    for trajs in [train_ensemble, test_ensemble]:
        process_trajs(data_dir, trajs)


if __name__ == '__main__':
    main()


