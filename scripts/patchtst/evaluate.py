import json
import logging
import os
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
import transformers
from gluonts.dataset.common import FileDataset

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.evaluation import (
    evaluate_forecasting_model,
    evaluate_mlm_model,
)
from dystformer.patchtst.pipeline import PatchTSTPipeline
from dystformer.utils import log_on_main, save_evaluation_results

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    checkpoint_path = cfg.eval.checkpoint_path
    log(f"Using checkpoint: {checkpoint_path}")
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    if os.path.exists(training_info_path):
        log(f"Training info file found at: {training_info_path}")
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
            train_config = training_info.get("train_config", None)
            if train_config is None:  # for backwards compatibility
                train_config = training_info.get("training_config", None)
    else:
        log(f"No training info file found at: {training_info_path}")
        train_config = None

    pipeline = PatchTSTPipeline.from_pretrained(
        mode=cfg.eval.mode, pretrain_path=checkpoint_path
    )
    model_config = dict(vars(pipeline.model.config))
    train_config = train_config or dict(cfg.train)
    # set floating point precision
    use_tf32 = train_config.get("tf32", False)
    log(f"use tf32: {use_tf32}")
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
        )
        use_tf32 = False

    rseed = train_config.get("seed", cfg.train.seed)
    log(f"Using SEED: {rseed}")
    transformers.set_seed(seed=rseed)

    context_length = model_config["context_length"]
    prediction_length = model_config["prediction_length"]
    channel_attention = model_config["channel_attention"]
    log(f"context_length: {context_length}")
    log(f"model prediction_length: {prediction_length}")
    log(f"eval prediction_length: {cfg.eval.prediction_length}")
    log(f"channel_attention: {channel_attention}")
    pipeline.model.eval()

    # get test data paths
    test_data_dir = os.path.expandvars(cfg.eval.data_path)
    test_data_dict = {}
    system_dirs = [d for d in Path(test_data_dir).iterdir() if d.is_dir()]

    for system_dir in system_dirs[: cfg.eval.num_systems]:
        system_name = system_dir.name
        system_files = list(system_dir.glob("*"))
        test_data_dict[system_name] = [
            FileDataset(path=Path(file_path), freq="h", one_dim_target=False)
            for file_path in system_files
            if file_path.is_file()
        ]

    log(f"Running evaluation on {list(test_data_dict.keys())}")

    test_datasets = {
        system_name: PatchTSTDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])]
            * len(test_data_dict[system_name]),
            context_length=context_length,
            prediction_length=cfg.eval.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            model_type=cfg.eval.mode,
            mode="test",
        )
        for system_name in test_data_dict
    }

    save_eval_results = partial(
        save_evaluation_results,
        metrics_save_dir=cfg.eval.metrics_save_dir,
        metrics_fname=cfg.eval.metrics_fname,
        overwrite=cfg.eval.overwrite,
        split_coords=cfg.eval.split_coords,
        verbose=cfg.eval.verbose,
    )
    log(f"Saving evaluation results to {cfg.eval.metrics_save_dir}")

    if cfg.eval.mode == "predict":
        parallel_sample_reduction_fn = {
            "mean": lambda x: np.mean(x, axis=0),
            "median": lambda x: np.median(x, axis=0),
        }.get(cfg.eval.parallel_sample_reduction, lambda x: x)

        predictions, contexts, labels, metrics = evaluate_forecasting_model(
            pipeline,
            test_datasets,
            batch_size=cfg.eval.batch_size,
            prediction_length=cfg.eval.prediction_length,
            limit_prediction_length=cfg.eval.limit_prediction_length,
            metric_names=cfg.eval.metric_names,
            return_predictions=True,
            return_contexts=True,
            return_labels=True,
            parallel_sample_reduction_fn=parallel_sample_reduction_fn,
            redo_normalization=True,
            eval_subintervals=[
                (0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)
            ],
        )

        if predictions is not None and contexts is not None:
            full_trajs = {}
            for system in predictions:
                if system not in contexts:
                    raise ValueError(f"System {system} not in contexts")
                # shape: (num_eval_windows*num_datasets, num_channels, context_length + prediction_length)
                full_trajs[system] = np.concatenate(
                    [contexts[system], predictions[system]], axis=2
                )
            save_eval_results(
                metrics,
                coords=full_trajs,
                coords_save_dir=cfg.eval.forecast_save_dir,
            )

        if labels is not None and contexts is not None:
            full_trajs = {}
            for system in labels:
                if system not in contexts:
                    raise ValueError(f"System {system} not in contexts")
                # shape: (num_eval_windows*num_datasets, num_channels, context_length + prediction_length)
                full_trajs[system] = np.concatenate(
                    [contexts[system], labels[system]], axis=2
                )
            save_eval_results(
                None,  # do not save metrics again
                coords=full_trajs,
                coords_save_dir=cfg.eval.labels_save_dir,
            )

    elif cfg.eval.mode == "pretrain":
        completions, processed_past_values, timestep_masks, metrics = (
            evaluate_mlm_model(
                pipeline,
                test_datasets,
                metric_names=cfg.eval.metric_names,
                batch_size=cfg.eval.batch_size,
                undo_normalization=False,
                return_completions=True,
                return_processed_past_values=True,
                return_masks=True,
            )
        )
        if completions is not None:
            save_eval_results(
                metrics,
                coords=completions,
                coords_save_dir=cfg.eval.completions_save_dir,
            )
        if processed_past_values is not None:
            save_eval_results(
                None,  # do not save metrics again
                coords=processed_past_values,
                coords_save_dir=cfg.eval.patch_input_save_dir,
            )
        if timestep_masks is not None:
            save_eval_results(
                None,  # do not save metrics again
                coords=timestep_masks,
                coords_save_dir=cfg.eval.timestep_masks_save_dir,
            )
    else:
        raise ValueError(f"Invalid eval mode: {cfg.eval.mode}")


if __name__ == "__main__":
    main()
