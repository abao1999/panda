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


def save_evaluation_results(
    metrics: dict[int, dict[str, dict[str, float]]] | None = None,
    metrics_metadata: dict[str, dict[str, Any]] | None = None,
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
        metrics_metadata: Dictionary containing metadata for the metrics.
            Keys are the quantity names, values are dictionaries containing metadata for each system.
                e.g. {"system_dims": {"system_name": 3}}
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
    if metrics is not None:
        for forecast_length, metric_dict in metrics.items():
            result_rows = [
                {"system": system, **metric_dict[system]} for system in metric_dict
            ]
            results_df = pd.DataFrame(result_rows)
            if metrics_metadata is not None:
                for quantity_name in metrics_metadata:
                    results_df[quantity_name] = results_df["system"].map(
                        metrics_metadata[quantity_name]
                    )
            curr_metrics_fname = (
                f"{metrics_fname or 'metrics'}_pred{forecast_length}.csv"
            )
            metrics_save_path = os.path.join(metrics_save_dir, curr_metrics_fname)
            logger.info(f"Saving metrics to: {metrics_save_path}")
            os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

            if os.path.isfile(metrics_save_path) and not overwrite:
                existing_df = pd.read_csv(metrics_save_path)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_csv(metrics_save_path, index=False)

    # save predictions, which is a dictionary mapping system names to prediction numpy arrays, to arrow files
    if coords_save_dir is not None and coords is not None:
        for system in coords:
            logger.info(f"{system}: {coords[system].shape}")

        os.makedirs(coords_save_dir, exist_ok=True)
        logger.info(
            f"Saving all valid sampled trajectories from {len(coords)} systems to arrow files within {coords_save_dir}",
        )

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
