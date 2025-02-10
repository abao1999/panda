import json
import logging
import os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable

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
    safe_standardize,
    save_evaluation_results,
)

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


def get_dim_from_dataset(dataset: FileDataset) -> int:
    """
    helper function to get system dimension from file dataset
    """
    return next(iter(dataset))["target"].shape[0]


def evaluate_chronos_forecast(
    pipeline: ChronosPipeline,
    systems: dict[str, ChronosDataset],
    batch_size: int,
    prediction_length: int,
    system_dims: dict[str, int],
    num_samples: int = 1,
    limit_prediction_length: bool = False,
    metric_names: list[str] | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
    parallel_sample_reduction_fn: Callable | None = None,
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
    **predict_kwargs: Any,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[int, dict[str, dict[str, float]]],
]:
    """
    Evaluate the model on each test system and save metrics.

    Args:
        pipeline: The Chronos Pipeline for evaluation.
        systems: A dictionary mapping system names to their respective ChronosDataset.
        batch_size: The batch size to use for evaluation.
        metric_names: Optional list of metric names to compute.
        parallel_sample_reduction: How to reduce the parallel samples over dim 0,
            only used if return_predictions is True
        return_predictions: Whether to return the predictions.
        return_contexts: Whether to return the contexts.
        return_labels: Whether to return the future values.
        system_dims: A dictionary mapping system names to their dimensions.

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
    assert isinstance(num_samples, int) and num_samples > 0, (
        "num_samples must be a positive integer"
    )
    system_predictions = {}
    system_contexts = {}
    system_labels = {}
    system_metrics = defaultdict(dict)

    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))

    if parallel_sample_reduction_fn is None:
        parallel_sample_reduction_fn = lambda x: x

    for system in tqdm(systems, desc="Forecasting..."):
        log(f"Evaluating {system}")
        dataset = systems[system]
        num_sys = len(dataset.datasets)
        dim = system_dims[system]
        predictions, labels, contexts, future_values = [], [], [], []

        # length of the dataset iterator is equal to len(dataset.datasets)*num_eval_windows
        # for each eval window style:
        # - sampled: num_eval_windows = num_test_instances
        # - rolling: num_eval_windows = (T - context_length - prediction_length) // window_stride + 1
        # - single: num_eval_windows = 1
        for batch in batcher(dataset, batch_size=batch_size):
            past_values, future_values = zip(
                *[(data["past_values"], data["future_values"]) for data in batch]
            )

            # num_samples = number of arrow files in this system's subdirectory
            # shape: (batch_size, context_length)
            past_batch = torch.cat(past_values, dim=0).to(pipeline.model.device).cpu()
            context = past_batch.numpy()

            # shape: (num_parallel_samples, batch_size, prediction_length)
            preds = (
                pipeline.predict(
                    past_batch,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    limit_prediction_length=limit_prediction_length,
                    **predict_kwargs,
                )
                .transpose(0, 1)
                .cpu()
                .numpy()
            )

            # shape: (batch_size, sampler_prediction_length)
            future_batch = torch.cat(future_values, dim=0).cpu().numpy()

            # Truncate predictions to match future_batch length if needed
            if preds.shape[-1] > future_batch.shape[-1]:
                preds = preds[..., : future_batch.shape[-1]]

            # standardize using the context from the past_batch
            if redo_normalization:
                future_batch = safe_standardize(future_batch, context=context)
                preds = safe_standardize(preds, context=context[None, :, :])
                context = safe_standardize(context)

            labels.append(future_batch)
            predictions.append(preds)
            contexts.append(context)

        # if parallel_sample_reduction_fn is None, then predictions shape is:
        # shape: (num_parallel_samples, num_systems*num_eval_windows, prediction_length, dim)
        # otherwise, predictions shape is:
        # shape: (num_systems*num_eval_windows, prediction_length, dim)
        predictions = (
            np.concatenate(predictions, axis=1)
            .reshape(num_samples, num_sys, dim, -1, prediction_length)
            .transpose(0, 1, 3, 4, 2)
            .reshape(num_samples, -1, prediction_length, dim)
        )
        predictions = parallel_sample_reduction_fn(predictions)
        labels = (
            np.concatenate(labels, axis=0)
            .reshape(num_sys, dim, -1, prediction_length)
            .transpose(0, 2, 3, 1)
            .reshape(-1, prediction_length, dim)
        )
        # shape: (num_systems*num_eval_windows, context_length, dim)
        contexts = (
            np.concatenate(contexts, axis=0)
            .reshape(num_sys, dim, -1, contexts[0].shape[-1])
            .transpose(0, 2, 3, 1)
            .reshape(-1, contexts[0].shape[-1], dim)
        )

        if metric_names is not None:
            assert all(start < prediction_length for start, _ in eval_subintervals), (
                "All start indices must be less than the prediction length"
            )
            for start, end in eval_subintervals:
                system_metrics[end - start][system] = compute_metrics(
                    predictions[:, start:end, :],
                    labels[:, start:end, :],
                    include=metric_names,
                    batch_axis=0,
                )

        if return_predictions:
            system_predictions[system] = predictions
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
    log(f"Using checkpoint: {checkpoint_path}")
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    if os.path.exists(training_info_path):
        log(f"Training info file found at: {training_info_path}")
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
            train_config = training_info.get("train_config", None)
            if train_config is None:  # for backwards compatibility
                train_config = training_info.get("training_config", None)
            dataset_config = training_info.get("dataset_config", None)
    else:
        log(f"No training info file found at: {training_info_path}")
        train_config = None
        dataset_config = None

    # init model for inference
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    pipeline = ChronosPipeline.from_pretrained(
        cfg.eval.checkpoint_path,
        device_map=cfg.eval.device,
        torch_dtype=torch_dtype,
    )
    pipeline.model.eval()

    model_config = dict(vars(pipeline.model.config))
    train_config = train_config or dict(cfg.train)
    dataset_config = dataset_config or dict(cfg)

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
    log(f"context_length: {context_length}")
    log(f"model prediction_length: {prediction_length}")
    log(f"eval prediction_length: {cfg.eval.prediction_length}")

    # get test data paths, collect all time series for each system in a dict (system_name -> list of FileDatasets)
    test_data_dir = os.path.expandvars(cfg.eval.data_path)
    test_data_dict = {}
    system_dirs = [d for d in Path(test_data_dir).iterdir() if d.is_dir()]
    for system_dir in random.sample(system_dirs, cfg.eval.num_systems):
        system_name = system_dir.name
        system_files = list(system_dir.glob("*"))
        test_data_dict[system_name] = [
            FileDataset(path=Path(file_path), freq="h", one_dim_target=False)
            for file_path in system_files
            if file_path.is_file()
        ]

    # for convenience, get system dimensions
    system_dims = {
        system_name: get_dim_from_dataset(test_data_dict[system_name][0])
        for system_name in test_data_dict
    }

    log(f"Running evaluation on {list(test_data_dict.keys())}")

    test_datasets = {
        system_name: ChronosDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])]
            * len(test_data_dict[system_name]),
            tokenizer=pipeline.tokenizer,
            context_length=cfg.chronos.context_length,
            prediction_length=cfg.eval.prediction_length,  # NOTE: should match the forecast prediction length
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
    log(f"Saving evaluation results to {cfg.eval.metrics_save_dir}")

    parallel_sample_reduction_fn = {
        "mean": lambda x: np.mean(x, axis=0),
        "median": lambda x: np.median(x, axis=0),
    }[cfg.eval.parallel_sample_reduction]

    predictions, contexts, labels, metrics = evaluate_chronos_forecast(
        pipeline,
        test_datasets,
        batch_size=cfg.eval.batch_size,
        prediction_length=cfg.eval.prediction_length,
        limit_prediction_length=cfg.eval.limit_prediction_length,
        metric_names=cfg.eval.metric_names,
        system_dims=system_dims,
        return_predictions=True,
        return_contexts=True,
        return_labels=True,
        parallel_sample_reduction_fn=parallel_sample_reduction_fn,
        redo_normalization=True,
        temperature=model_config["temperature"],
        top_k=model_config["top_k"],
        top_p=model_config["top_p"],
        eval_subintervals=[
            (0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)
        ],
    )

    log("Saving predictions...")
    if predictions is not None and contexts is not None:
        full_trajs = {}
        for system in predictions:
            if system not in contexts:
                raise ValueError(f"System {system} not in contexts")
            full_trajs[system] = np.concatenate(
                [contexts[system], predictions[system]], axis=1
            )
        save_eval_results(
            metrics,
            coords=full_trajs,
            coords_save_dir=None,  # cfg.eval.forecast_save_dir,
        )

    log("Saving labels...")
    if labels is not None and contexts is not None:
        full_trajs = {}
        for system in labels:
            if system not in contexts:
                raise ValueError(f"System {system} not in contexts")
            full_trajs[system] = np.concatenate(
                [contexts[system], labels[system]], axis=1
            )
        save_eval_results(
            None,  # do not save metrics again
            coords=full_trajs,
            coords_save_dir=None,  # cfg.eval.labels_save_dir,
        )


if __name__ == "__main__":
    main()
