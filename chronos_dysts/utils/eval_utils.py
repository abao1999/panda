# utils for evaluation script from original Chronos repo. To test if our modified chronos_dysts evaluation script
# Currently, we do not load from hugginface repo/dataset
# TODO: add functionality to load from huggingface repo/dataset

from typing import Iterable

import numpy as np
import torch
from gluonts.dataset.split import split
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.common import FileDataset
from pathlib import Path

import pyarrow.dataset as ds

from tqdm.auto import tqdm
from typing import Optional

# # TODO: fix this circular import thing, only used for typing though
# from chronos_dysts.pipeline import ChronosPipeline

def load_and_split_dataset(backtest_config: dict, verbose: Optional[bool] = False):
    """
    Takes in a config for a dyst system and loads the corresponding
    Arrow file into GluonTS FileDataset https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html 
    And then uses GluonTS split https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.split.html
    to generate test instances from windows of original timeseries
    NOTE: only works for separate dimension files, should be straightforward to extend
    """
    dyst_name = backtest_config["name"]
    filepath = backtest_config["data_filepath"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]

    if verbose:
        print(f"Loading {dyst_name} from {filepath}")
        print(f"Splitting timeseries by creating {num_rolls} non-overlapping windows")
        print(f"And using offset {offset} and prediction length {prediction_length}")

    gts_dataset = FileDataset(path=Path(filepath), freq="h") # TODO: consider other frequencies?

    # Split dataset for evaluation
    _, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return test_data


def generate_sample_forecasts(
    test_data_input: Iterable,
    pipeline: "ChronosPipeline",
    prediction_length: int,
    batch_size: int,
    num_samples: int,
    **predict_kwargs,
):
    """
    Generates forecast samples using GluonTS batcher to batch the generated instances from FileDataset
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

    print("Forecast Samples")
    print(forecast_samples)

    # Convert forecast samples into gluonts SampleForecast objects
    sample_forecasts = []
    for item, ts in zip(forecast_samples, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    return sample_forecasts
