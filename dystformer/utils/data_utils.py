import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

# for type hints
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
from dysts.systems import get_attractor_list
from gluonts.dataset import Dataset
from gluonts.dataset.arrow import ArrowWriter


def filter_dict(
    dictionary: Dict[Any, np.ndarray],
) -> Tuple[Dict[Any, np.ndarray], List[str]]:
    """
    Filter a dictionary by removing key-value pairs where the value is None.

    Args:
        dictionary (Dict[Any, np.ndarray]): The input dictionary to filter.

    Returns:
        Tuple[Dict[Any, np.ndarray], List[Any]]: A tuple containing:
            - The filtered dictionary with None values removed.
            - A list of keys that were excluded (i.e., had None values).
    """
    excluded_keys = [key for key, value in dictionary.items() if value is None]
    for key in excluded_keys:
        dictionary.pop(key)
    return dictionary, excluded_keys


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
    """
    Sample pairs from an arbitrary sequence. Returns iterator over Tuple[int, int]
    """
    num_total_pairs = size * (size - 1) // 2
    assert num_pairs <= num_total_pairs, "Cannot sample more pairs than unique pairs."
    sampled_pairs = (rng or np.random).choice(
        num_total_pairs, size=num_pairs, replace=False
    )
    all_pairs = list(combinations(range(size), 2))
    return (all_pairs[i] for i in sampled_pairs)


def get_system_filepaths(
    system_name: str, base_dir: Union[str, Path], split: str = "train"
) -> List[Path]:
    """
    Retrieve sorted filepaths for a given dynamical system.

    This function finds and sorts all Arrow files for a specified dynamical system
    within a given directory structure.

    Args:
        system_name (str): The name of the dynamical system.
        base_dir (Union[str, Path]): The base directory containing the data.
        split (str, optional): The data split to use (e.g., "train", "test"). Defaults to "train".

    Returns:
        List[Path]: A sorted list of Path objects for the Arrow files of the specified system.

    Raises:
        Exception: If the directory for the specified system does not exist.

    Note:
        The function assumes that the Arrow files are named with a numeric prefix
        (e.g., "1_T-1024.arrow") and sorts them based on this prefix.
    """
    dyst_dir = os.path.join(base_dir, split, system_name)
    if not os.path.exists(dyst_dir):
        raise Exception(f"Directory {dyst_dir} does not exist.")

    filepaths = sorted(
        list(Path(dyst_dir).glob("*.arrow")), key=lambda x: int(x.stem.split("_")[0])
    )
    return filepaths


### Functions to make basic affine maps ###
def pad_array(arr: np.ndarray, n2: int, m2: int) -> np.ndarray:
    """
    Pad an array to a target shape that is bigger than original shape
    """
    n1, m1 = arr.shape
    pad_rows, pad_cols = n2 - n1, m2 - m1
    if pad_rows < 0 or pad_cols < 0:
        raise ValueError(
            "Target dimensions must be greater than or equal to original dimensions."
        )
    return np.pad(
        arr, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
    )


def construct_basic_affine_map(
    n: int,
    m: int,
    kappa: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """
    Construct an affine map that sends (x, y, 1) -> (x, y, x + y)
    where x and y have lengths n and m respectively, and n <= m
    Args:
        n: driver system dimension
        m: response system dimension
        kappa: coupling strength, either a float or a list of floats
    Returns:
        A: the affine map matrix (2D array), block matrix (n + 2m) x (n + m + 1)
    """
    I_n = np.eye(n)  # n x n identity matrix
    I_m = np.eye(m)  # m x m identity matrix

    assert isinstance(
        kappa, (float, np.ndarray)
    ), "coupling strength kappa must be a float or a list of floats"

    if isinstance(kappa, float):
        bottom_block = np.hstack(
            [kappa * pad_array(I_n if n < m else I_m, m, n), I_m, np.zeros((m, 1))]
        )
    else:  # kappa is a list of floats
        k = min(n, m)
        assert len(kappa) == k, "coupling strength kappa must be of length min(n, m)"  # type: ignore
        bottom_block = np.hstack(
            [pad_array(np.diag(kappa), m, n), I_m, np.zeros((m, 1))]
        )

    top_block = np.hstack([I_n, np.zeros((n, m)), np.zeros((n, 1))])
    middle_block = np.hstack([np.zeros((m, n)), I_m, np.zeros((m, 1))])

    A = np.vstack([top_block, middle_block, bottom_block])
    return A
