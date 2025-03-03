import logging
import os
import time
from datetime import datetime
from functools import wraps
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from dysts.systems import get_attractor_list
from gluonts.dataset import Dataset
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.common import FileDataset


def get_dim_from_dataset(dataset: FileDataset) -> int:  # type: ignore
    """
    helper function to get system dimension from file dataset
    """
    return next(iter(dataset))["target"].shape[0]


def safe_standardize(
    arr: np.ndarray,
    epsilon: float = 1e-10,
    axis: int = -1,
    context: np.ndarray | None = None,
) -> np.ndarray:
    """
    Standardize the trajectories by subtracting the mean and dividing by the standard deviation

    Args:
        arr: The array to standardize
        epsilon: A small value to prevent division by zero
        axis: The axis to standardize along
        context: The context to use for standardization. If provided, use the context to standardize the array.

    Returns:
        The standardized array
    """
    # if no context is provided, use the array itself
    context = arr if context is None else context

    assert arr.ndim == context.ndim, (
        "arr and context must have the same num dims if context is provided"
    )
    assert axis < arr.ndim and axis >= -arr.ndim, (
        "invalid axis specified for standardization"
    )
    mean = np.nanmean(context, axis=axis, keepdims=True)
    std = np.nanstd(context, axis=axis, keepdims=True)
    std = np.where(std < epsilon, epsilon, std)
    return (arr - mean) / std


def demote_from_numpy(param: float | np.ndarray) -> float | list[float]:
    """
    Demote a float or numpy array to a float or list of floats
    Used for serializing parameters to json
    """
    if isinstance(param, np.ndarray):
        return param.tolist()
    return param


def dict_demote_from_numpy(
    param_dict: dict[str, float | np.ndarray],
) -> dict[str, float | list[float]]:
    """
    Demote a dictionary of parameters to a dictionary of floats or list of floats
    """
    return {k: demote_from_numpy(v) for k, v in param_dict.items()}


def timeit(logger: logging.Logger | None = None) -> Callable:
    """Decorator that measures and logs execution time of a function.

    Args:
        logger: Optional logger to use for timing output. If None, prints to stdout.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            if elapsed < 60:
                time_str = f"{elapsed:.2f} seconds"
            elif elapsed < 3600:
                time_str = f"{elapsed / 60:.2f} minutes"
            else:
                time_str = f"{elapsed / 3600:.2f} hours"

            msg = f"{func.__name__} took {time_str}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

            return result

        return wrapper

    return decorator


def split_systems(
    prop: float,
    seed: int,
    sys_class: str = "continuous",
    excluded_systems: List[str] = [],
) -> Tuple[List[str], List[str]]:
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
    ), "time_series must be a list of 1D numpy arrays or a 2D numpy array"

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


def accumulate_coords(
    filepaths: List[Path], one_dim_target: bool = False, num_samples: int | None = None
) -> np.ndarray:
    dyst_coords_samples = []
    for filepath in filepaths:
        if num_samples is not None and len(dyst_coords_samples) >= num_samples:
            break
        # create dataset by reading directly from filepath into FileDataset
        gts_dataset = FileDataset(
            path=Path(filepath),
            freq="h",
            one_dim_target=one_dim_target,
        )  # TODO: consider other frequencies?

        # extract the coordinates
        dyst_coords, metadata = stack_and_extract_metadata(
            gts_dataset,
        )

        dyst_coords_samples.append(dyst_coords)

    dyst_coords_samples = np.array(dyst_coords_samples)  # type: ignore
    return dyst_coords_samples


def process_trajs(
    base_dir: str,
    timeseries: dict[str, np.ndarray],
    split_coords: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    base_sample_idx: int = -1,
) -> None:
    """Saves each trajectory in timeseries ensemble to a separate directory"""
    for sys_name, trajectories in timeseries.items():
        if verbose:
            print(
                f"Processing trajectories of shape {trajectories.shape} for system {sys_name}"
            )

        system_folder = os.path.join(base_dir, sys_name)
        os.makedirs(system_folder, exist_ok=True)

        if not overwrite:
            for filename in os.listdir(system_folder):
                if filename.endswith(".arrow"):
                    sample_idx = int(filename.split("_")[0])
                    base_sample_idx = max(base_sample_idx, sample_idx)

        for i, trajectory in enumerate(trajectories):
            # very hacky, if there is only one trajectory, we can just use the base_sample_idx
            curr_sample_idx = base_sample_idx + i + (trajectories.shape[0] != 1)

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


def stack_and_extract_metadata(dataset: Dataset) -> Tuple[np.ndarray, Tuple[Any]]:
    """Utility for unpacking gluonts dataset into array and extracting metadata"""
    coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
    coordinates = np.stack(coords)
    if coordinates.ndim > 2:  # if not one_dim_target:
        coordinates = coordinates.squeeze()
    return coordinates, metadata


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


def process_dyst_name(
    dyst_name: str,
    base_dir: str,
    split: str,
    one_dim_target: bool,
    num_samples: int | None = None,
) -> Tuple[str, np.ndarray]:
    filepaths = get_system_filepaths(dyst_name, base_dir, split)
    dyst_coords_samples = []
    for filepath in filepaths[:num_samples]:
        # create dataset by reading directly from filepath into FileDataset
        gts_dataset = FileDataset(
            path=Path(filepath),
            freq="h",
            one_dim_target=one_dim_target,
        )
        # extract the coordinates
        dyst_coords, metadata = stack_and_extract_metadata(gts_dataset)
        dyst_coords_samples.append(dyst_coords)

    dyst_coords_samples = np.array(dyst_coords_samples)  # type: ignore
    print(dyst_coords_samples.shape)
    return dyst_name, dyst_coords_samples


def make_ensemble_from_arrow_dir(
    base_dir: str,
    split: str,
    dyst_names_lst: Optional[List[str]] = None,
    num_samples: int | None = None,
    one_dim_target: bool = False,
) -> Dict[str, np.ndarray]:
    ensemble = {}
    if dyst_names_lst is None:
        data_dir = os.path.join(base_dir, split)
        dyst_names_lst = [
            folder.name for folder in Path(data_dir).iterdir() if folder.is_dir()
        ]
    print(f"making ensemble from {split} split, with systems: {dyst_names_lst}")

    # Prepare arguments for multiprocessing
    args = [
        (dyst_name, base_dir, split, one_dim_target, num_samples)
        for dyst_name in dyst_names_lst
    ]

    # Use multiprocessing to process each dyst_name
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_dyst_name, args)

    # Collect results into the ensemble dictionary
    for dyst_name, dyst_coords_samples in results:
        ensemble[dyst_name] = dyst_coords_samples

    return ensemble
