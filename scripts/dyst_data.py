import os
import numpy as np

from pathlib import Path
from typing import List, Union, Dict, Iterable
from datetime import datetime
from dysts.base import get_attractor_list, make_trajectory_ensemble, init_cond_sampler
from dysts.utils import standardize_ts
from gluonts.dataset.arrow import ArrowWriter
import importlib
from typing import Optional

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
    start = datetime.now().strftime("%Y-%m-%d %H:%M")

    dataset = [
        {"start": start, "target": ts} for ts in time_series
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


def process_trajs(
    base_dir: str, timeseries: Dict[str, np.ndarray], selected_dims: Union[Iterable, str] = None) -> None:
    """Saves each trajectory in timeseries ensemble to a separate arrow file
    """
    for sys_name, trajectory in timeseries.items():
        if selected_dims is not None:
            for i in (range(trajectory.shape[-2]) if selected_dims == 'all' else selected_dims):
                path = os.path.join(base_dir, f"{sys_name}_dim-{i}.arrow")
                convert_to_arrow(path, trajectory[..., i, :])
        else: 
            path = os.path.join(base_dir, f"{sys_name}.arrow")
            convert_to_arrow(path, trajectory)


def main():

    rseed = 999
    num_periods = 5
    num_points = 1024
    num_ics = 2

    test, train = split_systems(0.3, seed=rseed)    
    test = train = ["Lorenz"]
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

        train_ensemble_list.append(train_ensemble)
        test_ensemble_list.append(test_ensemble)

    # Aggregate results (136 systems x 20 ics iter loop for each train and test ensemble aggregation)
    train_ensemble = {key: np.stack([d[key].T for d in train_ensemble_list], axis=0) for key in train_ensemble_list[0]}
    test_ensemble = {key: np.stack([d[key].T for d in test_ensemble_list], axis=0) for key in test_ensemble_list[0]}

    print("Saving timeseries to arrow files")
    for split, ensemble in [('train', train_ensemble), ('test', test_ensemble)]:
        data_dir = os.path.join(WORK_DIR, f'data/{split}')
        os.makedirs(data_dir, exist_ok=True)
        process_trajs(data_dir, ensemble, selected_dims='all')


if __name__ == '__main__':
    main()

