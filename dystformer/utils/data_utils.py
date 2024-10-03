import os
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
from dysts.systems import get_attractor_list
from gluonts.dataset import Dataset
from gluonts.dataset.arrow import ArrowWriter


def split_systems(
    prop: float,
    seed: int,
    sys_class: Optional[str] = "continuous",
    excluded_systems: List[str] = [],
):
    """
    Split the list of attractors into training and testing sets.
    if exclude_systems is provided, the systems in the list will be excluded
    """
    np.random.seed(seed)
    systems = get_attractor_list(sys_class=sys_class, exclude=excluded_systems)
    np.random.default_rng(seed).shuffle(systems)
    split = int(len(systems) * prop)
    return systems[:split], systems[split:]


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: Literal["lz4", "zstd"] = "lz4",
    split_coords: bool = False,
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and time_series.ndim == 2
    )

    # GluonTS requires this datetime format for reading arrow file
    start = datetime.now().strftime("%Y-%m-%d %H:%M")

    if split_coords:
        dataset = [{"start": start, "target": ts} for ts in time_series]
    else:
        dataset = [{"start": start, "target": time_series}]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=Path(path),
    )


def process_trajs(
    base_dir: str,
    timeseries: Dict[str, np.ndarray],
    split_coords: bool = False,
    verbose: bool = False,
) -> None:
    """Saves each trajectory in timeseries ensemble to a separate directory"""
    for sys_name, trajectories in timeseries.items():
        if verbose:
            print(
                f"Processing trajectories of shape {trajectories.shape} for system {sys_name}"
            )

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

            if trajectory.ndim == 1:
                trajectory = np.expand_dims(trajectory, axis=0)
            if verbose:
                print(
                    f"Saving {sys_name} trajectory {curr_sample_idx} with shape {trajectory.shape}"
                )

            path = os.path.join(
                system_folder, f"{curr_sample_idx}_T-{trajectory.shape[-1]}.arrow"
            )

            convert_to_arrow(path, trajectory, split_coords=split_coords)


## Utils for augmentations
def stack_and_extract_metadata(dataset: Dataset) -> Tuple[np.ndarray, Tuple[Any]]:
    """Utility for unpacking gluonts dataset into array and extracting metadata"""
    coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
    coordinates = np.stack(coords)
    if coordinates.ndim > 2:  # if not one_dim_target:
        coordinates = coordinates.squeeze()
    return coordinates, metadata


def sample_index_pairs(
    size: int, num_pairs: int, rng: Optional[np.random.Generator] = None
) -> Iterator:
    """Sample pairs from an arbitrary sequence
    TODO: add option to filter by dyst_name for sampled pairs?
    """
    num_total_pairs = size * (size - 1) // 2
    assert num_pairs <= num_total_pairs, "Cannot sample more pairs than unique pairs."
    sampled_pairs = (rng or np.random).choice(
        num_total_pairs, size=num_pairs, replace=False
    )
    all_pairs = list(combinations(range(size), 2))
    return (all_pairs[i] for i in sampled_pairs)


def filter_dict(d: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    # List to store the filtered out keys
    excluded_keys = []
    for key in list(d.keys()):
        if d[key] is None:  # or d[key].shape[0] < req_num_vals:
            excluded_keys.append(key)  # Collect the key
            del d[key]  # Remove the key from the dictionary
    return d, excluded_keys
