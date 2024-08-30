import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import os
import hydra

from tqdm.auto import tqdm

from collections import defaultdict

from gluonts.ev.metrics import SMAPE, MASE, RMSE, MeanWeightedSumQuantileLoss
# from gluonts.evaluation.metrics import smape, mase, rmse, wql
from gluonts.model.evaluation import evaluate_forecasts

from chronos_dysts.pipeline import ChronosPipeline
from chronos_dysts.utils import (
    load_and_split_dataset_from_arrow, 
    generate_sample_forecasts,
    average_nested_dict,
)


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    # create save path for evaluation metrics
    metrics_fname = cfg.eval.output_fname
    if metrics_fname is None:
        metrics_fname = f"{cfg.eval.split}_metrics.csv"
    metrics_path = os.path.join(cfg.eval.output_dir, metrics_fname)
    print("Saving metrics to: ", metrics_path)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # set floating point precision
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # default eval configs per dyst
    default_prediction_length = cfg.eval.prediction_length
    default_offset = cfg.eval.offset
    default_num_rolls = cfg.eval.num_rolls

    # get list of all dyst directories
    data_dir = os.path.join(cfg.eval.data_dir, cfg.eval.split)
    if not os.path.isdir(data_dir):
        raise Exception(f"Directory {data_dir} does not exist.")
    eval_dysts_names = []
    if data_dir is not None:
        eval_dysts_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # get custom dyst configs
    custom_dysts_config_lst = cfg.eval.custom_dysts
    custom_dysts_dict = {}
    print("Custom dysts configs: ", custom_dysts_config_lst)
    if custom_dysts_config_lst is not None:
        for d in custom_dysts_config_lst:
            dyst_name = d["name"]
            try:
                dyst_dir = d["path"]
            except KeyError:
                raise Exception(f"Custom dyst config for {dyst_name} does not contain a 'path' key.")
            if not os.path.isdir(dyst_dir):
                # continue
                raise Exception(f"Directory {dyst_dir} does not exist.")
            custom_dysts_dict[dyst_name] = {
                "path": dyst_dir,
                "prediction_length": d.get("prediction_length", default_prediction_length),
                "offset": d.get("offset", default_offset),
                "num_rolls": d.get("num_rolls", default_num_rolls),
                }

    eval_dysts_names.extend([d for d in custom_dysts_dict.keys() if d not in eval_dysts_names])
    print("Eval dyst dirs: ", eval_dysts_names)

    # Load model from checkpoint
    print(f"Loading Chronos checkpoint: {cfg.eval.model_id} onto device: {cfg.eval.device}")
    pipeline = ChronosPipeline.from_pretrained(
        cfg.eval.model_id,
        device_map=cfg.eval.device,
        torch_dtype=torch_dtype,
    )

    result_rows = []
    for dyst_name in tqdm(eval_dysts_names):
        dyst_dir = custom_dysts_dict.get(dyst_name, {}).get("path", os.path.join(data_dir, dyst_name))
        dyst_config = custom_dysts_dict.get(dyst_name, {"prediction_length": default_prediction_length, "offset": default_offset, "num_rolls": default_num_rolls})
        prediction_length = dyst_config["prediction_length"]
        offset = dyst_config["offset"]
        num_rolls = dyst_config["num_rolls"]
        print(f"Evaluating {dyst_name} from {dyst_dir} with prediction length {prediction_length} and offset {offset}")
        # get list of all dataset Arrow files associated with dyst_name
        # filepaths = list(Path(dyst_dir).glob("*.arrow"))
        filepaths = sorted(list(Path(dyst_dir).glob("*.arrow")), key=lambda x: int(x.stem.split("_")[0]))
        metrics_all_samples = defaultdict(lambda: defaultdict(list))
        for sample_idx, filepath in tqdm(enumerate(filepaths)): #, desc=f"evaluating metrics for all dataset files of {dyst_name}"):
            # load dataset test split from Arrow file
            logger.info(f"Loading sample index {sample_idx}, from {filepath}")
            test_data = load_and_split_dataset_from_arrow(
                prediction_length=prediction_length,
                offset=offset,
                num_rolls=num_rolls, 
                filepath=filepath,
            )

            # generate forecasts for all dimensions of a single sample instance
            logger.info(
                f"Generating forecasts for {dyst_name} sample {sample_idx} "
                f"with ({len(test_data.input)} time series)"
            )
            
            forecast_save_path = None
            if cfg.eval.save_forecasts_to_npy and cfg.eval.forecast_save_dir is not None:
                forecast_save_path = os.path.join(cfg.eval.forecast_save_dir, dyst_name, f"{filepath.stem}.npy")
            
            sample_forecasts = generate_sample_forecasts(
                test_data.input,
                pipeline=pipeline,
                prediction_length=prediction_length,
                batch_size=cfg.eval.batch_size,
                num_samples=cfg.eval.num_samples,
                limit_prediction_length=cfg.eval.limit_prediction_length,
                save_to_npy=cfg.eval.save_forecasts_to_npy,
                save_path=forecast_save_path,
                temperature=cfg.eval.temperature,
                top_k=cfg.eval.top_k,
                top_p=cfg.eval.top_p,
            )

            # TODO: add option to plot forecasts

            logger.info(f"Evaluating forecasts")

            # see gluonts metrics: https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ev/metrics.py
            # this returns a pandas DataFrame
            #    - if axis=None, aggregate along all dimensions
            #    - if axis=1, aggregate along time dimension, see: https://ts.gluon.ai/stable/_modules/gluonts/model/evaluation.html#evaluate_forecasts
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
                    axis=cfg.eval.agg_axis, # aggregation axis
                )
                .reset_index(drop=True)
                .to_dict(orient="records")
            )
            # metrics is list of dicts, each dict is a metric for a dimension
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
        metrics_all_samples = average_nested_dict(metrics_all_samples)
        # logger.info(metrics_all_samples)

        # aggregate metrics across all samples and dimensions of a dyst (average the errors across dimensions)
        if cfg.eval.agg_axis is None:
            assert len(metrics) == 1, "Expected only one dimension for axis=None aggregation"
            result_rows.append(
                {"dataset": dyst_name, "model": cfg.eval.model_id, **metrics_all_samples[0]}
            )
        # aggregate metrics across all samples of a dyst by dimension
        elif cfg.eval.agg_axis == 1:
            result_rows.extend(
                {"dataset": dyst_name, "dimension": dim_idx, "model": cfg.eval.model_id, **metrics_all_samples[dim_idx]}
                for dim_idx in range(len(metrics))
            )
        else:
            raise ValueError(f"Invalid aggregation axis: {cfg.eval.agg_axis}") # axis 0 and 2 are also allowed but we don't want them

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
    if os.path.isfile(metrics_path) and not cfg.eval.overwrite:
        existing_df = pd.read_csv(metrics_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(metrics_path, index=False)

    # TODO: embeddings, tokenizer_state = pipeline.embed(context)
    # TODO: get frequency from dataframe? Plot forecasts, interpret metrics, interpret encoder embeddings


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # TODO: need to log_on_main to activate logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    main()