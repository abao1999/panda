import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
import yaml
import os

from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts

from chronos_dysts.pipeline import ChronosPipeline
from chronos_dysts.utils import load_and_split_dataset, generate_sample_forecasts


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

    print("Pipeline")
    print(vars(pipeline))

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    print("Backtest configs")
    print(backtest_configs)

    result_rows = []
    for config in backtest_configs:
        print("config: ", config)
        dyst_name = config["name"]
        print("dyst name: ", dyst_name)
        prediction_length = config["prediction_length"]

        logger.info(f"Loading {dyst_name}")
        test_data = load_and_split_dataset(backtest_config=config)
        logger.info(
            f"Generating forecasts for {dyst_name} "
            f"({len(test_data.input)} time series)"
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

        logger.info(f"Evaluating forecasts for {dyst_name}")
        metrics = (
            evaluate_forecasts(
                sample_forecasts,
                test_data=test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        result_rows.append(
            {"dataset": dyst_name, "model": chronos_model_id, **metrics[0]}
        )

    # Save results to a CSV file
    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
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