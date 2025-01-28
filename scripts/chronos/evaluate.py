import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import transformers
from dysts.metrics import compute_metrics  # type: ignore
from gluonts.dataset.common import FileDataset
from gluonts.itertools import batcher
from gluonts.transform import LastValueImputation
from tqdm.auto import tqdm

from dystformer.chronos.dataset import ChronosDataset
from dystformer.chronos.pipeline import ChronosPipeline
from dystformer.utils import (
    log_on_main,
    save_evaluation_results,
)

logger = logging.getLogger(__name__)


def evaluate_chronos_forecast(
    pipeline: ChronosPipeline,
    systems: Dict[str, ChronosDataset],
    batch_size: int,
    prediction_length: int,
    limit_prediction_length: bool = False,
    metrics_names: Optional[List[str]] = None,
    parallel_sample_reduction: str = "none",
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
    **predict_kwargs,
) -> Tuple[
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Dict[str, Dict[str, float]],
]:
    """
    Evaluate the model on each test system and save metrics.

    Args:
        pipeline: The Chronos Pipeline for evaluation.
        systems: A dictionary mapping system names to their respective ChronosDataset.
        batch_size: The batch size to use for evaluation.
        metrics_names: Optional list of metric names to compute.
        parallel_sample_reduction: How to reduce the parallel samples over dim 0,
            only used if return_predictions is True
        return_predictions: Whether to return the predictions.
        return_contexts: Whether to return the contexts.
        return_labels: Whether to return the future values.
    Returns:
        A tuple containing:
        - system_predictions: A dictionary mapping system names to their predictions.
            Only returned if `return_predictions` is True.
        - system_contexts: A dictionary mapping system names to their contexts.
            Only returned if `return_contexts` is True.
        - system_labels: A dictionary mapping system names to their future values.
            Only returned if `return_labels` is True.
        - system_metrics: A nested dictionary containing computed metrics for each system.
    """
    system_predictions = {}
    system_contexts = {}
    system_labels = {}
    system_metrics = {system: {} for system in systems}

    parallel_sample_reduction_fn = {
        "mean": lambda x: np.mean(x, axis=0),
        "median": lambda x: np.median(x, axis=0),
    }.get(parallel_sample_reduction, lambda x: x)

    for system in tqdm(systems, desc="Forecasting..."):
        print(f"Evaluating {system}")
        dataset = systems[system]
        predictions, labels, contexts, future_values = [], [], [], []
        for batch in batcher(dataset, batch_size=batch_size):
            past_values, future_values = zip(
                *[(data["past_values"], data["future_values"]) for data in batch]
            )
            # shape: (dim * num_samples, 1, context_length) where num_samples is number of arrow files in this system's subdirectory
            past_batch = (
                torch.stack(past_values, dim=0).to(pipeline.model.device).squeeze(1)
            ).cpu()
            # shape: (dim * min(num_samples, batch_size), num_parallel_samples, prediction_length)
            preds = (
                pipeline.predict(
                    past_batch,
                    prediction_length=prediction_length,
                    num_samples=1,  # if None, defaults to chronos config
                    limit_prediction_length=limit_prediction_length,
                    **predict_kwargs,
                )
                .cpu()
                .numpy()
            ).transpose(1, 0, 2)

            context = past_batch.cpu().numpy()

            # shape: (dim * num_samples, sampler_prediction_length)
            future_batch = torch.stack(future_values, dim=0).squeeze(1).cpu().numpy()

            # Truncate predictions to match future_batch length if needed
            if preds.shape[-1] > future_batch.shape[-1]:
                preds = preds[..., : future_batch.shape[-1]]

            if redo_normalization:
                # compute loc and scale from past_batch
                loc = np.nanmean(context, axis=1)
                scale = np.nanstd(context, axis=1)
                scale = np.where(scale < 1e-10, 1e-10, scale)
                loc = np.expand_dims(loc, axis=1)
                scale = np.expand_dims(scale, axis=1)
                future_batch = (future_batch - loc) / scale
                context = (context - loc) / scale

            print(
                f"future_batch shape: {future_batch.shape}, preds shape: {preds.shape}"
            )
            print(f"context shape: {context.shape}")
            labels.append(future_batch)
            predictions.append(preds)
            contexts.append(context)

        predictions = np.concatenate(predictions, axis=1)
        print(predictions.shape)
        labels = np.concatenate(labels, axis=0)
        print(labels.shape)
        contexts = np.concatenate(contexts, axis=0)
        print(contexts.shape)

        if metrics_names is not None:
            system_metrics[system] = compute_metrics(
                np.squeeze(predictions),
                labels,
                include=metrics_names,  # type: ignore
            )

        # shape: (num_parallel_samples, num_windows*num_datasets, prediction_length, num_channels)
        # or (num_windows*num_datasets, prediction_length, num_channels) if parallel_sample_reduction is not none
        if return_predictions:
            system_predictions[system] = parallel_sample_reduction_fn(predictions)
        if return_contexts:
            system_contexts[system] = contexts
        if return_labels:
            system_labels[system] = labels

    return (
        system_predictions if return_predictions else None,
        system_contexts if return_contexts else None,
        system_labels if return_labels else None,
        system_metrics,
    )


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    checkpoint_path = cfg.eval.checkpoint_path
    log_on_main(f"Using checkpoint: {checkpoint_path}", logger)
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    if os.path.exists(training_info_path):
        log_on_main(f"Training info file found at: {training_info_path}", logger)
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
            train_config = training_info.get("train_config", None)
            if train_config is None:  # for backwards compatibility
                train_config = training_info.get("training_config", None)
            dataset_config = training_info.get("dataset_config", None)
    else:
        log_on_main(f"No training info file found at: {training_info_path}", logger)
        train_config = None
        dataset_config = None

    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    pipeline = ChronosPipeline.from_pretrained(
        cfg.eval.checkpoint_path,
        device_map=cfg.eval.device,
        torch_dtype=torch_dtype,
    )
    model_config = dict(vars(pipeline.model.config))
    train_config = train_config or dict(cfg.train)
    dataset_config = dataset_config or dict(cfg)
    # set floating point precision
    use_tf32 = train_config.get("tf32", False)
    log_on_main(f"use tf32: {use_tf32}", logger)
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        use_tf32 = False

    rseed = train_config.get("seed", cfg.train.seed)
    log_on_main(f"Using SEED: {rseed}", logger)
    transformers.set_seed(seed=rseed)

    context_length = model_config["context_length"]
    prediction_length = model_config["prediction_length"]
    log_on_main(f"context_length: {context_length}", logger)
    log_on_main(f"prediction_length: {prediction_length}", logger)
    pipeline.model.eval()

    # get test data paths
    test_data_dir = os.path.expandvars(cfg.eval.data_path)
    test_data_dict = {}

    # Get all system directories and randomly sample num_systems of them
    system_dirs = [d for d in Path(test_data_dir).iterdir() if d.is_dir()]

    for i, system_dir in enumerate(system_dirs):
        if i > cfg.eval.num_systems:
            break
        system_name = system_dir.name
        system_files = list(system_dir.glob("*"))
        test_data_dict[system_name] = [
            FileDataset(path=Path(file_path), freq="h", one_dim_target=False)
            for file_path in system_files
            if file_path.is_file()
        ]

    log_on_main(f"Running evaluation on {list(test_data_dict.keys())}", logger)

    test_datasets = {
        system_name: ChronosDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])]
            * len(test_data_dict[system_name]),
            tokenizer=pipeline.tokenizer,
            context_length=cfg.chronos.context_length,
            prediction_length=cfg.chronos.prediction_length,
            min_past=cfg.min_past,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            model_type=cfg.chronos.model_type,
            imputation_method=LastValueImputation()
            if cfg.chronos.model_type == "causal"
            else None,
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
    logger.info(f"Saving evaluation results to {cfg.eval.metrics_save_dir}")

    predictions, contexts, labels, metrics = evaluate_chronos_forecast(
        pipeline,
        test_datasets,
        batch_size=cfg.eval.batch_size,
        prediction_length=cfg.eval.prediction_length,
        limit_prediction_length=cfg.eval.limit_prediction_length,
        metrics_names=cfg.eval.metrics_names,
        return_predictions=True,
        return_contexts=True,
        return_labels=True,
        parallel_sample_reduction="mean",
        redo_normalization=False,
        temperature=model_config["temperature"],
        top_k=model_config["top_k"],
        top_p=model_config["top_p"],
    )
    window_indices = None

    print("Saving predictions...")
    if predictions is not None and contexts is not None:
        full_trajs = {}
        for system in predictions:
            if system not in contexts:
                raise ValueError(f"System {system} not in contexts")
            full_trajs[system] = np.concatenate(
                [contexts[system], predictions[system]], axis=1
            )
            print(full_trajs[system].shape)
        save_eval_results(
            metrics,
            coords=full_trajs,
            window_indices=window_indices,
            coords_save_dir=None,  # cfg.eval.forecast_save_dir,
        )

    print("Saving labels...")
    if labels is not None and contexts is not None:
        full_trajs = {}
        for system in labels:
            if system not in contexts:
                raise ValueError(f"System {system} not in contexts")
            full_trajs[system] = np.concatenate(
                [contexts[system], labels[system]], axis=1
            )
            print(full_trajs[system].shape)
        save_eval_results(
            metrics,
            coords=full_trajs,
            window_indices=window_indices,
            coords_save_dir=None,  # cfg.eval.labels_save_dir,
        )


if __name__ == "__main__":
    main()
