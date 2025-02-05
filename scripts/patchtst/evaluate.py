import json
import logging
import os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
import torch
import transformers
from dysts.metrics import compute_metrics
from gluonts.dataset.common import FileDataset
from gluonts.itertools import batcher
from tqdm.auto import tqdm

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTST
from dystformer.utils import (
    log_on_main,
    safe_standardize,
    save_evaluation_results,
)

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


def evaluate_mlm_model(
    model: PatchTST,
    systems: dict[str, PatchTSTDataset],
    batch_size: int,
    metrics_names: list[str] | None = None,
    undo_normalization: bool = False,
    return_completions: bool = False,
    return_processed_past_values: bool = False,
    return_masks: bool = False,
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
    assert model.mode == "pretrain", "Model must be in pretrain mode"
    system_completions = {}
    system_processed_past_values = {}
    system_timestep_masks = {}
    system_metrics = defaultdict(dict)

    context_length = model.config.context_length
    # random_mask_ratio = model.config.random_mask_ratio

    for system in tqdm(systems, desc="Evaluating MLM pretrain model"):
        dataset = systems[system]  # IterableDataset
        log(f"Evaluating {system}")
        all_completions = []
        all_processed_past_values = []
        all_timestep_masks = []
        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            past_values = [data["past_values"] for data in batch]

            # past_batch = torch.from_numpy(np.stack(past_values)).to(model.device)
            past_batch = torch.stack(past_values, dim=0).to(model.device)

            # note past_observed_mask is None because we don't have any missing values in the training data
            completions_output = model.model.generate_completions(
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

            # transpose back to (batch_size, num_channels, num_timesteps)
            completions = completions.transpose(0, 2, 1)
            processed_past_values = processed_past_values.transpose(0, 2, 1)

            if metrics_names is not None:
                eval_metrics = compute_metrics(
                    completions,
                    processed_past_values,
                    include=metrics_names,  # type: ignore
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
            system_completions[system] = full_completion
        if return_processed_past_values:
            full_processed_past_values = np.concatenate(
                all_processed_past_values, axis=0
            )
            system_processed_past_values[system] = full_processed_past_values
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
    model: PatchTST,
    systems: dict[str, PatchTSTDataset],
    batch_size: int,
    prediction_length: int,
    limit_prediction_length: bool = False,
    metric_names: list[str] | None = None,
    parallel_sample_reduction_fn: Callable | None = None,
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
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
        systems: A dictionary mapping system names to their respective PatchTSTDataset.
        batch_size: The batch size to use for evaluation.
        prediction_length: The length of the predictions to make.
        limit_prediction_length: Whether to limit the prediction length to the prediction length.
        metric_names: Optional list of metric names to compute.
        parallel_sample_reduction_fn: How to reduce the parallel samples over dim 0
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
    assert model.mode == "predict", "Model must be in predict mode"
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
            past_batch = torch.stack(past_values, dim=0).to(model.device)
            context = past_batch.cpu().numpy()

            # shape: (num_parallel_samples, batch size, prediction_length, num_channels)
            preds = (
                model.predict(
                    past_batch,
                    prediction_length=prediction_length,
                    limit_prediction_length=limit_prediction_length,
                )
                .cpu()
                .numpy()
            ).transpose(1, 0, 2, 3)

            # shape: (batch size, sampler_prediction_length, num_channels)
            future_batch = torch.stack(future_values, dim=0).cpu().numpy()

            # Truncate predictions to match future_batch length if needed
            if preds.shape[2] > future_batch.shape[1]:
                preds = preds[..., : future_batch.shape[1], :]

            # standardize using stats from the past_batch
            if redo_normalization:
                future_batch = safe_standardize(future_batch, context=context, axis=1)
                preds = safe_standardize(preds, context=context[None, :, :], axis=2)
                context = safe_standardize(context, axis=1)

            labels.append(future_batch)
            predictions.append(preds)
            contexts.append(context)

        # if num_parallel_reduction_fn is None, the shape is:
        # shape: (num_parallel_samples, num_eval_windows*num_datasets, prediction_length, num_channels)
        # otherwise, the shape is:
        # shape: (num_eval_windows*num_datasets, prediction_length, num_channels)
        predictions = np.concatenate(predictions, axis=1)
        predictions = parallel_sample_reduction_fn(predictions)
        # shape: (num_eval_windows*num_datasets, context_length, num_channels)
        labels = np.concatenate(labels, axis=0)
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
                )

        # if parallel_sample_reduction_fn is not None, the shape is:
        # shape: (num_eval_windows*num_datasets, prediction_length, num_channels)
        # otherwise, the shape is:
        # shape: (num_parallel_samples, num_eval_windows*num_datasets, prediction_length, num_channels)
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

    model = PatchTST.from_pretrained(
        mode=cfg.eval.mode,
        pretrain_path=checkpoint_path,
        device=cfg.eval.device,
    )
    model_config = dict(vars(model.config))
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

    # use_quadratic_embedding may not exist for older checkpoints
    use_quadratic_embedding = dataset_config.get(
        "use_quadratic_embedding", cfg.use_quadratic_embedding
    )
    context_length = model_config["context_length"]
    prediction_length = model_config["prediction_length"]
    log(f"use_quadratic_embedding: {use_quadratic_embedding}")
    log(f"context_length: {context_length}")
    log(f"model prediction_length: {prediction_length}")
    log(f"prediction_length: {cfg.eval.prediction_length}")
    model.eval()

    # get test data paths
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
            model,
            test_datasets,
            batch_size=cfg.eval.batch_size,
            prediction_length=cfg.eval.prediction_length,
            limit_prediction_length=cfg.eval.limit_prediction_length,
            metric_names=cfg.eval.metrics_names,
            return_predictions=True,
            return_contexts=True,
            return_labels=True,
            parallel_sample_reduction_fn=parallel_sample_reduction_fn,
            redo_normalization=True,
            eval_subintervals=[
                (0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)
            ],
        )
        window_indices = None
        # # get the indices of each prediction window for each timeseries in each system
        # window_indices = rolling_prediction_window_indices(
        #     test_data_dict,
        #     cfg.eval.window_stride,
        #     context_length,
        #     prediction_length,
        # )

        if predictions is not None and contexts is not None:
            full_trajs = {}
            for system in predictions:
                if system not in contexts:
                    raise ValueError(f"System {system} not in contexts")
                # shape: (num_eval_windows*num_datasets, context_length + prediction_length, num_channels)
                full_trajs[system] = np.concatenate(
                    [contexts[system], predictions[system]], axis=1
                )
            save_eval_results(
                metrics,
                coords=full_trajs,
                window_indices=window_indices,
                coords_save_dir=cfg.eval.forecast_save_dir,
            )

        if labels is not None and contexts is not None:
            full_trajs = {}
            for system in labels:
                if system not in contexts:
                    raise ValueError(f"System {system} not in contexts")
                # shape: (num_eval_windows*num_datasets, context_length + prediction_length, num_channels)
                full_trajs[system] = np.concatenate(
                    [contexts[system], labels[system]], axis=1
                )
            save_eval_results(
                None,  # do not save metrics again
                coords=full_trajs,
                window_indices=window_indices,
                coords_save_dir=cfg.eval.labels_save_dir,
            )

    elif cfg.eval.mode == "pretrain":
        completions, processed_past_values, timestep_masks, metrics = (
            evaluate_mlm_model(
                model,
                test_datasets,
                metrics_names=cfg.eval.metrics_names,
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
