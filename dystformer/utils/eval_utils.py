"""
utils for evaluation scripts
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import FileDataset
from gluonts.dataset.split import TestData, split

from dystformer.utils import process_trajs

logger = logging.getLogger(__name__)


def left_pad_and_stack_multivariate(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Left pad a list of multivariate time series tensors to the same length and stack them.
    Used in pipeline, if given context is a list of tensors.
    """
    max_len = max(c.shape[0] for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 2
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def rolling_prediction_window_indices(
    datasets: dict[str, list],
    window_stride: int,
    context_length: int,
    prediction_length: int,
) -> dict[str, list[list[int]]]:
    """
    Get the indices of individual windows for each timeseries in each system
    in the stacked prediction window array from a rolling sampler.

    This can handle multiple timeseries for each system, possible with different lengths

    For each system with stack prediction window array W of shape (..., num_windows*num_datasets, ...),
    return a list of tuples where each tuple contains the indices of the windows of its
    predictions e.g. W[..., shapes[system_name][i], ...] extracts the windows for the
    i-th timeseries of that system.

    Args:
        datasets: Dictionary mapping system names to lists of datasets
        window_stride: Stride length between consecutive windows
        context_length: Length of context window
        prediction_length: Length of prediction window
        window_dim: Dimension along which to count windows (default: 0)

    Returns:
        Dictionary mapping system names to shape of the unpacked window arrays
    """
    indices = {}
    for system_name, timeseries in datasets.items():
        indices[system_name] = []
        for i, dataset in enumerate(timeseries):
            ts_shape = next(iter(dataset))["target"].shape
            assert len(ts_shape) == 2, "Target must be 2D"
            T = ts_shape[-1]
            windows = (T - context_length - prediction_length) // window_stride + 1
            indices[system_name].append(list(range(i * windows, (i + 1) * windows)))
    return indices


def sampled_prediction_window_indices(
    datasets: dict[str, list],
    num_samples: int,
) -> dict[str, list[list[int]]]:
    """
    Get the indices of individual windows for each timeseries in each system
    in the stacked prediction window array from a sampled sampler.

    This can handle multiple timeseries for each system, possibly with different lengths.

    For each system with stack prediction window array W of shape (..., num_samples*num_datasets, ...),
    return a list of tuples where each tuple contains the indices of the windows of its
    predictions e.g. W[..., shapes[system_name][i], ...] extracts the windows for the
    i-th timeseries of that system.

    NOTE: Assumes that the sample-style window sampler sampled a fixed number of
    windows for each timeseries

    Args:
        datasets: Dictionary mapping system names to lists of datasets
        num_samples: Number of samples to draw for each timeseries
        context_length: Length of context window
        prediction_length: Length of prediction window

    Returns:
        Dictionary mapping system names to shape of the unpacked window arrays
    """
    indices = {
        system_name: [
            list(range(i * num_samples, (i + 1) * num_samples))
            for i in range(len(datasets[system_name]))
        ]
        for system_name in datasets
    }
    return indices


def save_evaluation_results(
    metrics: dict[str, dict[str, float]],
    window_indices: dict[str, list[list[int]]] | None = None,
    window_dim: int = 0,
    coords: dict[str, np.ndarray] | None = None,
    metrics_save_dir: str = "results",
    metrics_fname: str | None = None,
    overwrite: bool = False,
    coords_save_dir: str | None = None,
    split_coords: bool = False,
    verbose: bool = False,
):
    """
    Save prediction metrics and optionally forecast trajectories.

    Args:
        coords: Dictionary mapping system names to coordinate numpy arrays.
        metrics: Nested dictionary containing computed metrics for each system.
        window_indices: Optional dictionary mapping system names to indices of the windows of its predictions.
        window_dim: Dimension along which to count windows (default: 0)
        metrics_save_dir: Directory to save metrics to.
        metrics_fname: Name of the metrics file to save.
        overwrite: Whether to overwrite an existing metrics file
                    AND also overwrite any existing arrow files when saving coords
        coords_save_dir: Directory to save forecast trajectories to.
        split_coords: Whether to split the coordinates by dimension
        verbose: Whether to print verbose output.

    This function performs two main tasks:
    1. Saves evaluation metrics to a CSV file, appending to existing file if present.
    2. If specified in eval_cfg, saves forecast trajectories as arrow files.
    """
    result_rows = [{"system": system, **metrics[system]} for system in metrics]
    results_df = pd.DataFrame(result_rows)

    metrics_fname = f"{metrics_fname or 'metrics'}.csv"
    metrics_save_path = os.path.join(metrics_save_dir, metrics_fname)
    logger.info(f"Saving metrics to: {metrics_save_path}")
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

    if os.path.isfile(metrics_save_path) and not overwrite:
        existing_df = pd.read_csv(metrics_save_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(metrics_save_path, index=False)

    # save predictions, which is a dictionary mapping system names to prediction numpy arrays, to arrow files
    if coords_save_dir is not None and coords is not None:
        logger.info(f"forecast dysts: {coords.keys()}")
        for system in coords:
            logger.info(f"{system}: {coords[system].shape}")

        os.makedirs(coords_save_dir, exist_ok=True)
        logger.info(
            f"Saving all valid sampled trajectories from {len(coords)} systems to arrow files within {coords_save_dir}",
        )

        # regroup the predictions by the window indices before writing to arrow files
        # currently each prediction array (for each system) has shape:
        # [num_parallel_samples, sum(num_windows for each timeseries), prediction_length, num_channels]
        # or if a reduction was applied to the parallel sample dim:
        # [sum(num_windows for each timeseries), prediction_length, num_channels]
        if window_indices is not None:
            for system in coords:
                regrouped_coords = [
                    np.take(coords[system], indices, axis=window_dim)
                    for indices in window_indices[system]
                ]

                # regrouped_predictions is a ragged list of arrays of shape:
                # (num_windows_for_this_timeseries, prediction_length, num_channels)
                coords[system] = regrouped_coords  # type: ignore

        process_trajs(
            coords_save_dir,
            coords,
            split_coords=split_coords,
            overwrite=overwrite,
            verbose=verbose,
        )


### OLD CHRONOS UTILS (TODO: delete eventually, once we refactor Chronos code more) ###
def left_pad_and_stack_1D(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Left pad a list of 1D tensors to the same length and stack them.
    Used in pipeline, if given context is a list of tensors.
    """
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def load_and_split_dataset_from_arrow(
    prediction_length: int,
    offset: int,
    num_rolls: int,
    filepath: str | Path,
    one_dim_target: bool = False,
    verbose: bool = False,
) -> TestData:
    """
    Directly loads Arrow file into GluonTS FileDataset
        https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html
    And then uses GluonTS split to generate test instances from windows of original timeseries
        https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.split.html

    Load and split a dataset from an Arrow file, applies split to generate test instances
    Returns:
      ``TestData`` object.
            Elements of a ``TestData`` object are pairs ``(input, label)``, where
            ``input`` is input data for models, while ``label`` is the future
            ground truth that models are supposed to predict.
    """
    # TODO: load selected dimensions for dyst system config, so we can have separate forecast for each univariate trajectory
    if verbose:
        print("loading: ", filepath)
        print(f"Splitting timeseries by creating {num_rolls} non-overlapping windows")
        print(f"And using offset {offset} and prediction length {prediction_length}")

    gts_dataset = FileDataset(
        path=Path(filepath), freq="h", one_dim_target=one_dim_target
    )

    # Split dataset for evaluation
    _, test_template = split(gts_dataset, offset=offset)

    # see Gluonts split documentation: https://ts.gluon.ai/v0.11.x/_modules/gluonts/dataset/split.html#TestTemplate.generate_instances
    # https://ts.gluon.ai/v0.11.x/api/gluonts/gluonts.dataset.split.html#gluonts.dataset.split.TestData
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return test_data


def average_nested_dict(
    data: dict[Any, dict[Any, list[float]]],
) -> dict[Any, dict[Any, float]]:
    return {
        outer_key: {
            inner_key: sum(values) / len(values)
            for inner_key, values in outer_dict.items()
        }
        for outer_key, outer_dict in data.items()
    }
