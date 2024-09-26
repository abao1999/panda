"""
utils for Chronos Pipeline (evaluation) and evaluation scripts
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from git import Union
from gluonts.dataset.common import FileDataset
from gluonts.dataset.split import TestData, split
from gluonts.itertools import batcher
from gluonts.model.forecast import Forecast, SampleForecast
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from dystformer.chronos.pipeline import ChronosPipeline


def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor:
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
    filepath: Union[str, Path],
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
        path=Path(filepath), freq="h"
    )  # TODO: consider other frequencies?

    # Split dataset for evaluation
    _, test_template = split(gts_dataset, offset=offset)

    # see Gluonts split documentation: https://ts.gluon.ai/v0.11.x/_modules/gluonts/dataset/split.html#TestTemplate.generate_instances
    # https://ts.gluon.ai/v0.11.x/api/gluonts/gluonts.dataset.split.html#gluonts.dataset.split.TestData
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return test_data


def generate_sample_forecasts(
    test_data_input: Iterable,
    pipeline: "ChronosPipeline",
    prediction_length: int,
    batch_size: int,
    num_samples: int,
    limit_prediction_length: bool = True,
    save_path: Optional[str] = None,
    **predict_kwargs,
) -> Iterable[Forecast]:
    """
    Generates forecast samples using GluonTS batcher to batch the test instances generated from FileDataset
    Returns Forecast object https://ts.gluon.ai/stable/api/gluonts/gluonts.model.forecast.html#gluonts.model.forecast.Forecast
    """
    forecast_samples = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]
        forecast_samples.append(
            pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                limit_prediction_length=limit_prediction_length,
                **predict_kwargs,
            ).numpy()
        )
    forecast_samples = np.concatenate(forecast_samples)
    print("Forecast Samples shape: ", forecast_samples.shape)
    if save_path is not None:
        print(f"Saving forecast samples to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, forecast_samples)

    sample_forecasts = []
    for item, ts in zip(forecast_samples, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            # see https://ts.gluon.ai/stable/api/gluonts/gluonts.model.forecast.html#gluonts.model.forecast.SampleForecast
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    return sample_forecasts


def average_nested_dict(
    data: Dict[Any, Dict[Any, List[float]]],
) -> Dict[Any, Dict[Any, float]]:
    return {
        outer_key: {
            inner_key: sum(values) / len(values)
            for inner_key, values in outer_dict.items()
        }
        for outer_key, outer_dict in data.items()
    }
