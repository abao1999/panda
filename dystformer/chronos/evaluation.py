from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from dysts.metrics import compute_metrics  # type: ignore
from gluonts.itertools import batcher
from tqdm.auto import tqdm

from dystformer.chronos.dataset import ChronosDataset
from dystformer.chronos.pipeline import ChronosBoltPipeline, ChronosPipeline
from dystformer.utils import safe_standardize


def evaluate_chronos_forecast(
    pipeline: ChronosPipeline | ChronosBoltPipeline,
    systems: dict[str, ChronosDataset],
    batch_size: int,
    prediction_length: int,
    system_dims: dict[str, int],
    metric_names: list[str] | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
    parallel_sample_reduction_fn: Callable | None = None,
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
    prediction_kwargs: dict | None = None,
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
    system_predictions = {}
    system_contexts = {}
    system_labels = {}
    system_metrics = defaultdict(dict)
    prediction_kwargs = prediction_kwargs or {}
    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))

    if parallel_sample_reduction_fn is None:
        parallel_sample_reduction_fn = lambda x: x

    pbar = tqdm(systems, desc="Forecasting...")
    for system in pbar:
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

            # shape: (batch_size, context_length)
            past_batch = torch.cat(past_values, dim=0).to(pipeline.model.device).cpu()
            context = past_batch.numpy()

            # shape: (num_parallel_samples, batch_size, prediction_length)
            predict_args = {
                "context": past_batch,
                "prediction_length": prediction_length,
                **prediction_kwargs,
            }
            preds = pipeline.predict(**predict_args).transpose(0, 1).cpu().numpy()
            num_samples = preds.shape[0]

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
            system_predictions[system] = predictions.transpose(0, 2, 1)
        if return_contexts:
            system_contexts[system] = contexts.transpose(0, 2, 1)
        if return_labels:
            system_labels[system] = labels.transpose(0, 2, 1)

        pbar.set_postfix({"system": system, "num systems": num_sys})

    return (
        system_predictions if return_predictions else None,
        system_contexts if return_contexts else None,
        system_labels if return_labels else None,
        system_metrics,
    )
