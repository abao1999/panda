import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
import yaml
import os
from tqdm.auto import tqdm

from collections import defaultdict

# TODO: attractor error metrics that consider forecasts for all dimensions jointly
from gluonts.ev.metrics import SMAPE, MASE, RMSE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts

from chronos_dysts.pipeline import ChronosPipeline
from chronos_dysts.utils import (
    load_and_split_dataset_from_arrow, 
    generate_sample_forecasts,
    average_nested_dict,
)


app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    config_path: Path,
    metrics_path: Path,
    chronos_model_id: str = "amazon/chronos-t5-small",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    print("Config path: ", config_path)
    print("Metrics (save) path: ", metrics_path)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # Load Chronos
    pipeline = ChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch_dtype,
    )

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    data_dir = backtest_configs["data_dir"]
    dysts_configs = backtest_configs["dysts"]
    print("Dysts configs: ", dysts_configs)

    result_rows = []
    # for each dynamical system
    for dyst_config in dysts_configs:
        # get dyst config
        logger.info("config: ", dyst_config)
        dyst_name = dyst_config["name"]
        prediction_length = dyst_config["prediction_length"]
        # check if data directory exists
        dyst_data_dir = os.path.join(data_dir, dyst_name)
        if not os.path.exists(dyst_data_dir):
            continue
            # raise Exception(f"Directory {dyst_data_dir} does not exist.")
        print(f"Evaluating {dyst_name} with prediction length {prediction_length}")

        # get list of all dataset Arrow files associated with dyst_name
        filepaths = list(Path(dyst_data_dir).glob("*.arrow"))
        metrics_all_samples = defaultdict(lambda: defaultdict(list))
        for sample_idx, filepath in tqdm(enumerate(filepaths), desc=f"evaluating metrics for all dataset files of {dyst_name}"):
            # load dataset test split from Arrow file
            logger.info(f"Loading sample index {sample_idx}, from {filepath}")
            test_data = load_and_split_dataset_from_arrow(dyst_config, filepath)

            # generate forecasts for all dimensions of a single sample instance
            logger.info(
                f"Generating forecasts for {dyst_name} sample {sample_idx} "
                f"with ({len(test_data.input)} time series)"
            )
            sample_forecasts = generate_sample_forecasts(
                test_data.input,
                pipeline=pipeline,
                prediction_length=prediction_length,
                batch_size=batch_size,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # TODO: add option to plot forecasts

            logger.info(f"Evaluating forecasts")

            # see gluonts metrics: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ev/metrics.py
            # this returns a pandas DataFrame
            #    - need axis=1, aggregate along time dimension, see: https://ts.gluon.ai/stable/_modules/gluonts/model/evaluation.html#evaluate_forecasts
            #    - Can also aggregate test_data before metrics call so we can use gluonts api with axis=0 
            metrics = (
                evaluate_forecasts(
                    sample_forecasts,
                    test_data=test_data,
                    metrics=[
                        SMAPE(),
                        MASE(),
                        RMSE(),
                        MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                    ],
                    batch_size=5000,
                    axis=1, # aggregate along time dimension
                )
                .reset_index(drop=True)
                .to_dict(orient="records")
            )
            # metrics is list of dicts, each dict is a metric for a dimension
            print(len(metrics)) # TODO: check if dimension is correct, matched dysts dimension

            # Verify that all dictionaries in metrics have the same keys
            keys = metrics[0].keys()
            if not all(m.keys() == keys for m in metrics):
                raise ValueError("Not all dictionaries (per dim) in metrics list have the same keys.")
                    
            # Combine metrics into metrics_all_samples using defaultdicts
            for dim_idx, metrics_per_dim in enumerate(metrics):
                for metric_name, metric_value in metrics_per_dim.items():
                    metrics_all_samples[dim_idx][metric_name].append(metric_value)
    
        # aggregate metrics across all samples of each dyst dim
        #   i.e. each dim has its own dict that contains averaged values across all samples for that dim
        print(metrics_all_samples)
        metrics_all_samples = average_nested_dict(metrics_all_samples)
        print(metrics_all_samples)
        # aggregate metrics across all samples of a dyst by dimension
        for dim_idx in range(len(metrics)):
            result_rows.append(
                {"dataset": dyst_name, "dimension": dim_idx, "model": chronos_model_id, **metrics_all_samples[dim_idx]}
            )

    # Save results to a CSV file
    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {
                "sMAPE[0.5]": "sMAPE",
                "MASE[0.5]": "MASE",
                "RMSE[mean]": "RMSE",
                "mean_weighted_sum_quantile_loss": "WQL"
            },
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    results_df.to_csv(metrics_path, index=False)

    # TODO: embeddings, tokenizer_state = pipeline.embed(context)
    # TODO: get frequency from dataframe? Plot forecasts, interpret metrics, interpret encoder embeddings


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()
    print(typer.get_command(app))