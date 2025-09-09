from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from dysts.metrics import compute_metrics  # type: ignore
from gluonts.itertools import batcher
from panda.patchtst.dataset import TimeSeriesDataset
from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils import safe_standardize
from tqdm import tqdm


# TODO: test this and compute metrics for mlms, distributional metrics
def evaluate_mlm_model(
    pipeline: PatchTSTPipeline,
    systems: dict[str, TimeSeriesDataset],
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
    """
    past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
    Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
    in `[0, 1]`:
    """
    assert pipeline.mode == "pretrain", "Model must be in pretrain mode"
    system_completions = {}
    system_processed_past_values = {}
    system_timestep_masks = {}
    system_metrics = defaultdict(dict)

    context_length = pipeline.model.config.context_length

    # TODO: de-aggregate the "systems"-level metrics to separate each sample (unique parameter perturbation)
    for system in tqdm(systems, desc="Evaluating MLM pretrain model"):
        dataset = systems[system]  # IterableDataset
        all_completions = []
        all_processed_past_values = []
        all_timestep_masks = []
        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            past_values = [data["past_values"] for data in batch]

            past_batch = torch.stack(past_values, dim=0).to(pipeline.device)

            # note past_observed_mask is None because we don't have any missing values in the training data
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

            # get the masks
            patch_size = completions_output.completions.shape[-1]
            if completions_output.mask is None:
                raise ValueError("Mask is None")
            # shape: (batch_size, num_channels, num_patches)
            patch_mask = completions_output.mask.detach().cpu().numpy()
            # shape: (batch_size, num_channels, num_timesteps)
            timestep_mask = np.repeat(patch_mask, repeats=patch_size, axis=2)

            # get the patched past values after instance normalization
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

            # NOTE: important! completions and processed_past_values must be of shape:
            # (batch_size, num_timesteps, num_channels) in order for compute_metrics to work as intended
            # TODO: check this in debugging
            if metric_names is not None:
                eval_metrics = compute_metrics(
                    completions,
                    processed_past_values,
                    include=metric_names,  # type: ignore
                )
                # compute running average of metrics over batches
                for metric, value in eval_metrics.items():
                    system_metrics[system][metric] += (
                        value - system_metrics[system][metric]
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
            full_processed_past_values = np.concatenate(
                all_processed_past_values, axis=0
            )
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


def evaluate_forecasting_model(
    pipeline: PatchTSTPipeline,
    systems: dict[str, TimeSeriesDataset],
    batch_size: int,
    prediction_length: int,
    metric_names: list[str] | None = None,
    parallel_sample_reduction_fn: Callable | None = None,
    channel_sampler: Callable | None = None,
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
    """
    Evaluate the model on each test system and save metrics.

    Args:
        model: The PatchTST model to evaluate.
        systems: A dictionary mapping system names to their respective TimeSeriesDataset.
        batch_size: The batch size to use for evaluation.
        prediction_length: The length of the predictions to make.
        limit_prediction_length: Whether to limit the prediction length to the prediction length.
        metric_names: Optional list of metric names to compute.
        parallel_sample_reduction_fn: How to reduce the parallel samples over dim 0
        channel_sampler: callable which subsamples channels from context via the model.predict method
        return_predictions: Whether to return the predictions.
        return_contexts: Whether to return the contexts.
        return_labels: Whether to return the future values.
        redo_normalization: Whether to redo the normalization of the future values.

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

    pbar = tqdm(systems, desc="Forecasting...")
    for system in pbar:
        dataset = systems[system]
        num_sys = len(dataset.datasets)
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
            # shape: (batch_size, context_length, num_channels)
            past_batch = torch.stack(past_values, dim=0).to(pipeline.device)

            # shape: (num_parallel_samples, batch size, prediction_length, num_channels)
            # shit changes when using a channel sampler. notably, the batch dim, but it doesnt
            # matter if one just needs to aggregate over that to get metrics
            preds = (
                pipeline.predict(
                    past_batch, prediction_length=prediction_length, **prediction_kwargs
                )
                .transpose(0, 1)
                .cpu()
                .numpy()
            )
            context = past_batch.cpu().numpy()

            # shape: (batch size, sampler_prediction_length, num_channels)
            future_batch = torch.stack(future_values, dim=0).cpu().numpy()

            # Truncate predictions to match future_batch length if needed
            if preds.shape[2] > future_batch.shape[1]:
                preds = preds[..., : future_batch.shape[1], :]

            # if channel sampler is used, the preds batch size and num_channels changes
            # reflect the changes in the other tensors as well
            if channel_sampler is not None:
                future_batch = channel_sampler(
                    torch.from_numpy(future_batch), resample_inds=False
                ).numpy()
                context = channel_sampler(
                    torch.from_numpy(context), resample_inds=False
                ).numpy()

            # standardize using stats from the past_batch
            if redo_normalization:
                preds = safe_standardize(preds, context=context[None, :, :], axis=2)
                future_batch = safe_standardize(future_batch, context=context, axis=1)
                context = safe_standardize(context, axis=1)

            labels.append(future_batch)
            predictions.append(preds)
            contexts.append(context)

        # if num_parallel_reduction_fn is None, the shape is:
        # shape: (num_parallel_samples, num_eval_windows * num_datasets, prediction_length, num_channels)
        # otherwise, the shape is:
        # shape: (num_eval_windows * num_datasets, prediction_length, num_channels)
        predictions = np.concatenate(predictions, axis=1)
        predictions = parallel_sample_reduction_fn(predictions)
        labels = np.concatenate(labels, axis=0)

        # shape: (num_eval_windows * num_datasets, context_length, num_channels)
        contexts = np.concatenate(contexts, axis=0)

        # evaluate metrics for multiple forecast lengths on user-specified subintervals
        # as well as the full prediction length interval
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

        # shape: (num_eval_windows * num_datasets, num_channels, prediction_length or context_length)
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
