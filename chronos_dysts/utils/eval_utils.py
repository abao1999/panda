"""
utils for Chronos Pipeline (evaluation) and evaluation scripts
"""
from typing import Iterable

import numpy as np
import torch
from gluonts.dataset.split import split, TestData
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast, Forecast
from gluonts.dataset.common import FileDataset
from pathlib import Path

from tqdm.auto import tqdm
from typing import Optional, Any, List, Dict


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
        filepath: str, 
        verbose: Optional[bool] = False
) -> TestData:
    """
    Directly loads Arrow file into GluonTS FileDataset
        https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html 
    And then uses GluonTS split to generate test instances from windows of original timeseries
        https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.split.html
    """
    # TODO: load selected dimensions for dyst system config, so we can have separate forecast for each univariate trajectory
    if verbose:
        print("filepath: ", filepath)
        print(f"Splitting timeseries by creating {num_rolls} non-overlapping windows")
        print(f"And using offset {offset} and prediction length {prediction_length}")

    print("loading ", filepath)
    gts_dataset = FileDataset(path=Path(filepath), freq="h") # TODO: consider other frequencies?

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
                **predict_kwargs,
            ).numpy()
        )
    forecast_samples = np.concatenate(forecast_samples)
    print("Forecast Samples shape: ", forecast_samples.shape)

    # Convert forecast samples into gluonts SampleForecast objects
    sample_forecasts = []
    for item, ts in zip(forecast_samples, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            # see https://ts.gluon.ai/stable/api/gluonts/gluonts.model.forecast.html#gluonts.model.forecast.SampleForecast
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    return sample_forecasts


def average_nested_dict(data: Dict[Any, Dict[Any, List[float]]]) -> Dict[Any, Dict[Any, float]]:
    return {
        outer_key: {
            inner_key: sum(values) / len(values) for inner_key, values in outer_dict.items()
        }
        for outer_key, outer_dict in data.items()
    }