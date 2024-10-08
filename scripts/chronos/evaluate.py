"""

TODO: why is so much coderewritten without making use of the chronos dataset which has a test sampler?
"""
import logging
import os
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from gluonts.ev.metrics import MASE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts
from tqdm.auto import tqdm

from dystformer.chronos.pipeline import ChronosPipeline
from dystformer.utils import (
    average_nested_dict,
    generate_sample_forecasts,
    load_and_split_dataset_from_arrow,
)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    metrics_fname = cfg.eval.output_fname or f"{cfg.eval.split}_metrics.csv"
    metrics_path = os.path.join(cfg.eval.output_dir, metrics_fname)
    print("Saving metrics to: ", metrics_path)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    default_prediction_length = cfg.eval.prediction_length
    default_offset = cfg.eval.offset
    default_num_rolls = cfg.eval.num_rolls

    data_dir = os.path.join(cfg.eval.data_dir, cfg.eval.split)
    if not os.path.isdir(data_dir):
        raise Exception(f"Directory {data_dir} does not exist.")
    eval_dysts_names = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    custom_dysts_config_lst = cfg.eval.custom_dysts
    custom_dysts_dict = {}
    print("Custom dysts configs: ", custom_dysts_config_lst)
    if custom_dysts_config_lst:
        for d in custom_dysts_config_lst:
            dyst_name = d["name"]
            dyst_dir = d.get("path")
            if not dyst_dir or not os.path.isdir(dyst_dir):
                raise Exception(f"Directory {dyst_dir} does not exist.")
            custom_dysts_dict[dyst_name] = {
                "path": dyst_dir,
                "prediction_length": d.get(
                    "prediction_length", default_prediction_length
                ),
                "offset": d.get("offset", default_offset),
                "num_rolls": d.get("num_rolls", default_num_rolls),
            }

    eval_dysts_names.extend(
        [d for d in custom_dysts_dict.keys() if d not in eval_dysts_names]
    )
    print("Eval dyst dirs: ", eval_dysts_names)

    print(
        f"Loading Chronos checkpoint: {cfg.eval.model_id} onto device: {cfg.eval.device}"
    )
    pipeline = ChronosPipeline.from_pretrained(
        cfg.eval.model_id,
        device_map=cfg.eval.device,
        torch_dtype=torch_dtype,
    )

    result_rows = []
    for dyst_name in tqdm(eval_dysts_names):
        dyst_dir = custom_dysts_dict.get(dyst_name, {}).get(
            "path", os.path.join(data_dir, dyst_name)
        )
        dyst_config = custom_dysts_dict.get(
            dyst_name,
            {
                "prediction_length": default_prediction_length,
                "offset": default_offset,
                "num_rolls": default_num_rolls,
            },
        )
        prediction_length = dyst_config["prediction_length"]
        offset = dyst_config["offset"]
        num_rolls = dyst_config["num_rolls"]
        print(
            f"Evaluating {dyst_name} from {dyst_dir} with prediction length {prediction_length} and offset {offset}"
        )

        filepaths = sorted(
            list(Path(dyst_dir).glob("*.arrow")),
            key=lambda x: int(x.stem.split("_")[0]),
        )
        metrics_all_samples = defaultdict(lambda: defaultdict(list))
        for sample_idx, filepath in tqdm(enumerate(filepaths)):
            logger.info(f"Loading sample index {sample_idx}, from {filepath}")
            test_data = load_and_split_dataset_from_arrow(
                prediction_length=prediction_length,
                offset=offset,
                num_rolls=num_rolls,
                filepath=filepath,
            )

            logger.info(
                f"Generating forecasts for {dyst_name} sample {sample_idx} with ({len(test_data.input)} time series)"
            )

            forecast_save_path = None
            if cfg.eval.forecast_save_dir:
                forecast_save_path = os.path.join(
                    cfg.eval.forecast_save_dir, dyst_name, f"{filepath.stem}.npy"
                )
                os.makedirs(os.path.dirname(forecast_save_path), exist_ok=True)

            sample_forecasts = generate_sample_forecasts(
                test_data.input,
                pipeline=pipeline,
                prediction_length=prediction_length,
                batch_size=cfg.eval.batch_size,
                num_samples=cfg.eval.num_samples,
                limit_prediction_length=cfg.eval.limit_prediction_length,
                save_path=forecast_save_path,
                temperature=cfg.eval.temperature,
                top_k=cfg.eval.top_k,
                top_p=cfg.eval.top_p,
            )

            logger.info("Evaluating forecasts")

            metrics = []
            if test_data.input:
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
                        axis=cfg.eval.agg_axis,
                    )
                    .reset_index(drop=True)
                    .to_dict(orient="records")
                )

            keys = metrics[0].keys()
            if not all(m.keys() == keys for m in metrics):
                raise ValueError(
                    "Not all dictionaries (per dim) in metrics list have the same keys."
                )

            for dim_idx, metrics_per_dim in enumerate(metrics):
                for metric_name, metric_value in metrics_per_dim.items():
                    metrics_all_samples[dim_idx][metric_name].append(metric_value)

        metrics_dict = {k: dict(v) for k, v in metrics_all_samples.items()}
        metrics_all_samples = average_nested_dict(metrics_dict)

        if cfg.eval.agg_axis is None:
            assert (
                len(metrics_all_samples) == 1
            ), "Expected only one dimension for axis=None aggregation"
            result_rows.append(
                {
                    "dataset": dyst_name,
                    "model": cfg.eval.model_id,
                    **metrics_all_samples[0],
                }
            )
        elif cfg.eval.agg_axis == 1:
            result_rows.extend(
                {
                    "dataset": dyst_name,
                    "dimension": dim_idx,
                    "model": cfg.eval.model_id,
                    **metrics_all_samples[dim_idx],
                }
                for dim_idx in range(len(metrics_all_samples))
            )
        else:
            raise ValueError(f"Invalid aggregation axis: {cfg.eval.agg_axis}")

    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {
                "sMAPE[0.5]": "sMAPE",
                "MASE[0.5]": "MASE",
                "RMSE[mean]": "RMSE",
                "mean_weighted_sum_quantile_loss": "WQL",
            },
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    if os.path.isfile(metrics_path) and not cfg.eval.overwrite:
        existing_df = pd.read_csv(metrics_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    main()
