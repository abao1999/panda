from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch
from dysts.metrics import compute_metrics  # type: ignore
from gluonts.itertools import batcher
from panda.dataset import (
    MultivariateTimeSeriesDataset,
    UnivariateTimeSeriesDataset,
)
from panda.utils import safe_standardize
from tqdm import tqdm


def evaluate_univariate_forecasting_model(
    pipeline: Any,
    systems: dict[str, UnivariateTimeSeriesDataset],
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

    for system in tqdm(systems, desc="Forecasting..."):
        dataset = systems[system]
        num_sys = len(dataset.datasets)
        dim = system_dims[system]
        predictions, labels, contexts, future_values = [], [], [], []

        for batch in batcher(dataset, batch_size=batch_size):
            past_values, future_values = zip(
                *[(data["past_values"], data["future_values"]) for data in batch]
            )

            past_batch = torch.cat(past_values, dim=0).to(pipeline.model.device).cpu()
            context = past_batch.numpy()

            predict_args = {
                "context": past_batch,
                "prediction_length": prediction_length,
                **prediction_kwargs,
            }
            preds = pipeline.predict(**predict_args).transpose(0, 1).cpu().numpy()
            num_samples = preds.shape[0]

            future_batch = torch.cat(future_values, dim=0).cpu().numpy()

            if preds.shape[-1] > future_batch.shape[-1]:
                preds = preds[..., : future_batch.shape[-1]]

            if redo_normalization:
                future_batch = safe_standardize(future_batch, context=context)
                preds = safe_standardize(preds, context=context[None, :, :])
                context = safe_standardize(context)

            labels.append(future_batch)
            predictions.append(preds)
            contexts.append(context)

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
        contexts = (
            np.concatenate(contexts, axis=0)
            .reshape(num_sys, dim, -1, contexts[0].shape[-1])
            .transpose(0, 2, 3, 1)
            .reshape(-1, contexts[0].shape[-1], dim)
        )

        if metric_names is not None:
            assert all(start < prediction_length for start, _ in eval_subintervals)
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

    return (
        system_predictions if return_predictions else None,
        system_contexts if return_contexts else None,
        system_labels if return_labels else None,
        system_metrics,
    )


def evaluate_multivariate_forecasting_model(
    pipeline: Any,
    systems: dict[str, MultivariateTimeSeriesDataset],
    batch_size: int,
    prediction_length: int,
    metric_names: list[str] | None = None,
    parallel_sample_reduction_fn: Callable | None = None,
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
    prediction_kwargs: dict | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[int, dict[str, dict[str, float]]],
]:
    assert pipeline.mode == "predict", "Model must be in predict mode"
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

    for system in tqdm(systems, desc="Forecasting..."):
        dataset = systems[system]
        num_sys = len(dataset.datasets)
        predictions, labels, contexts, future_values = [], [], [], []

        for batch in batcher(dataset, batch_size=batch_size):
            past_values, future_values = zip(
                *[(data["past_values"], data["future_values"]) for data in batch]
            )
            past_batch = torch.stack(past_values, dim=0).to(pipeline.device)

            preds = (
                pipeline.predict(
                    past_batch, prediction_length=prediction_length, **prediction_kwargs
                )
                .transpose(0, 1)
                .cpu()
                .numpy()
            )
            context = past_batch.cpu().numpy()

            future_batch = torch.stack(future_values, dim=0).cpu().numpy()
            if preds.shape[2] > future_batch.shape[1]:
                preds = preds[..., : future_batch.shape[1], :]

            if redo_normalization:
                preds = safe_standardize(preds, context=context[None, :, :], axis=2)
                future_batch = safe_standardize(future_batch, context=context, axis=1)
                context = safe_standardize(context, axis=1)

            labels.append(future_batch)
            predictions.append(preds)
            contexts.append(context)

        predictions = np.concatenate(predictions, axis=1)
        predictions = parallel_sample_reduction_fn(predictions)
        labels = np.concatenate(labels, axis=0)
        contexts = np.concatenate(contexts, axis=0)

        if metric_names is not None:
            assert all(start < prediction_length for start, _ in eval_subintervals)
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

    return (
        system_predictions if return_predictions else None,
        system_contexts if return_contexts else None,
        system_labels if return_labels else None,
        system_metrics,
    )


def evaluate_multivariate_mlm_model(
    pipeline: Any,
    systems: dict[str, MultivariateTimeSeriesDataset],
    batch_size: int,
    metric_names: list[str] | None = None,
    undo_normalization: bool = False,
    return_processed_past_values: bool = False,
    return_masks: bool = False,
    return_completions: bool = False,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[int, dict[str, dict[str, float]]],
]:
    assert pipeline.mode == "pretrain", "Model must be in pretrain mode"
    system_completions: dict[str, np.ndarray] = {}
    system_processed_past_values: dict[str, np.ndarray] = {}
    system_timestep_masks: dict[str, np.ndarray] = {}
    system_metrics: dict[str, dict[str, float]] = defaultdict(dict)

    context_length = pipeline.model.config.context_length

    for system in tqdm(systems, desc="Evaluating MLM pretrain model"):
        dataset = systems[system]
        all_completions = []
        all_processed_past_values = []
        all_timestep_masks = []
        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            past_values = [data["past_values"] for data in batch]
            past_batch = torch.stack(past_values, dim=0).to(pipeline.device)

            completions_output = pipeline.model.generate_completions(
                past_batch, past_observed_mask=None
            )
            completions = (
                completions_output.completions.reshape(
                    past_batch.shape[0], past_batch.shape[-1], -1
                )
                .detach()
                .cpu()
                .numpy()
                .transpose(0, 2, 1)
            )

            patch_size = completions_output.completions.shape[-1]
            if completions_output.mask is None:
                raise ValueError("Mask is None")
            patch_mask = completions_output.mask.detach().cpu().numpy()
            timestep_mask = np.repeat(patch_mask, repeats=patch_size, axis=2)

            if completions_output.patched_past_values is None:
                raise ValueError("Patched past values are None")
            processed_past_values = (
                completions_output.patched_past_values.reshape(
                    past_batch.shape[0], past_batch.shape[-1], -1
                )
                .detach()
                .cpu()
                .numpy()
                .transpose(0, 2, 1)
            )

            if undo_normalization:
                if completions_output.loc is None or completions_output.scale is None:
                    raise ValueError("Loc or scale is None")
                loc = completions_output.loc.detach().cpu().numpy()
                scale = completions_output.scale.detach().cpu().numpy()
                completions = completions * scale + loc
                processed_past_values = processed_past_values * scale + loc

            if metric_names is not None:
                eval_metrics = compute_metrics(
                    completions,
                    processed_past_values,
                    include=metric_names,  # type: ignore
                )
                for metric, value in eval_metrics.items():
                    system_metrics[system][metric] += (
                        value - system_metrics[system].get(metric, 0.0)
                    ) / (i + 1)

            if return_completions:
                all_completions.append(completions)
            if return_processed_past_values:
                all_processed_past_values.append(processed_past_values)
            if return_masks:
                all_timestep_masks.append(timestep_mask)

        if return_completions:
            full_completion = np.concatenate(all_completions, axis=0)
            system_completions[system] = full_completion.transpose(0, 2, 1)
        if return_processed_past_values:
            full_processed_past_values = np.concatenate(all_processed_past_values, axis=0)
            system_processed_past_values[system] = full_processed_past_values.transpose(
                0, 2, 1
            )
        if return_masks:
            full_timestep_masks = np.concatenate(all_timestep_masks, axis=0)
            system_timestep_masks[system] = full_timestep_masks

    return (
        system_completions if return_completions else None,
        system_processed_past_values if return_processed_past_values else None,
        system_timestep_masks if return_masks else None,
        {context_length: system_metrics},
    )


