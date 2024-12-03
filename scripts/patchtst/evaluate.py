import logging
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import batcher
from tqdm.auto import tqdm

from dystformer.augmentations import (
    FixedDimensionDelayEmbeddingTransform,
    QuadraticEmbeddingTransform,
)
from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTST
from dystformer.utils import (
    log_on_main,
    save_evaluation_results,
)

logger = logging.getLogger(__name__)


def evaluate_mlm_model(
    model: PatchTST,
    systems: Dict[str, PatchTSTDataset],
    batch_size: int,
    metrics: Optional[List[str]] = None,
    undo_normalization: bool = False,
    return_completions: bool = False,
    return_processed_past_values: bool = False,
    return_masked_past_values: bool = False,
    num_systems: Optional[int] = None,
) -> Tuple[
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Dict[str, Dict[str, float]],
]:
    """
    past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
    Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
    in `[0, 1]`:
    """
    assert model.mode == "pretrain", "Model must be in pretrain mode"
    system_completions = {}
    system_processed_past_values = {}
    system_masked_past_values = {}
    system_metrics = {system: defaultdict(float) for system in systems}

    for idx, system in tqdm(enumerate(systems), desc="Evaluating MLM pretrain model"):
        if num_systems is not None and idx >= num_systems:
            break

        dataset = systems[system]  # IterableDataset
        log_on_main(f"Evaluating {system}", logger)
        all_completions = []
        all_processed_past_values = []
        all_masked_past_values = []
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
            breakpoint()

            # processed_past_values = past_batch.detach().cpu().numpy()
            masked_past_values = None

            if undo_normalization:
                if completions_output.loc is None or completions_output.scale is None:
                    raise ValueError("Loc or scale is None")
                loc = completions_output.loc.detach().cpu().numpy()
                scale = completions_output.scale.detach().cpu().numpy()
                completions = completions * scale + loc
                processed_past_values = processed_past_values * scale + loc

            completions = completions.transpose(0, 2, 1)
            processed_past_values = processed_past_values.transpose(0, 2, 1)

            # eval_metrics = compute_metrics(completions, past_batch.cpu().numpy())

            # NOTE: need newest dyst version for this
            # # compute running average of metrics over batches
            # for metric, value in eval_metrics.items():
            #     system_metrics[system][metric] += (
            #         value - system_metrics[system][metric]
            #     ) / (i + 1)

            if return_completions:
                all_completions.append(completions)
            if return_processed_past_values:
                all_processed_past_values.append(processed_past_values)
            if return_masked_past_values:
                all_masked_past_values.append(masked_past_values)

        if return_completions:
            full_completion = np.concatenate(all_completions, axis=1)
            system_completions[system] = full_completion
        if return_processed_past_values:
            full_processed_past_values = np.concatenate(
                all_processed_past_values, axis=1
            )
            system_processed_past_values[system] = full_processed_past_values
        if return_masked_past_values:
            full_masked_past_values = np.concatenate(all_masked_past_values, axis=1)
            system_masked_past_values[system] = full_masked_past_values

    # convert defaultdicts to regular dicts
    system_metrics = {
        system: dict(metrics) for system, metrics in system_metrics.items()
    }

    return (
        system_completions if return_completions else None,
        system_processed_past_values if return_processed_past_values else None,
        system_masked_past_values if return_masked_past_values else None,
        system_metrics,
    )


def evaluate_forecasting_model(
    model: PatchTST,
    systems: Dict[str, PatchTSTDataset],
    batch_size: int,
    prediction_length: int,
    limit_prediction_length: bool = False,
    metrics: Optional[List[str]] = None,
    parallel_sample_reduction: str = "none",
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
) -> Tuple[
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Dict[str, Dict[str, float]],
]:
    """
    Evaluate the model on each test system and save metrics.

    Args:
        model: The PatchTST model to evaluate.
        systems: A dictionary mapping system names to their respective PatchTSTDataset.
        batch_size: The batch size to use for evaluation.
        metrics: Optional list of metric names to compute.
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
    assert model.mode == "predict", "Model must be in predict mode"
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
            past_batch = torch.stack(past_values, dim=0).to(model.device)

            # shape: (num_parallel_samples, batch_size, prediction_length, num_channels)
            preds = (
                model.predict(
                    past_batch,
                    prediction_length=prediction_length,
                    limit_prediction_length=limit_prediction_length,
                )
                .cpu()
                .numpy()
                .transpose(1, 0, 2, 3)
            )

            context = past_batch.cpu().numpy()

            # shape: (batch_size, sampler_prediction_length, num_channels)h
            future_batch = torch.stack(future_values, dim=0).cpu().numpy()
            # Truncate predictions to match future_batch length if needed
            if preds.shape[2] > future_batch.shape[1]:
                preds = preds[..., : future_batch.shape[1], :]

            if redo_normalization:
                # compute loc and scale from past_batch
                loc = context.mean(axis=1)
                scale = context.std(axis=1)
                loc = np.expand_dims(loc, axis=1)
                scale = np.expand_dims(scale, axis=1)
                future_batch = (future_batch - loc) / scale
                # preds = (preds - loc) / scale
                context = (context - loc) / scale

            labels.append(future_batch)
            predictions.append(preds)
            contexts.append(context)
        # num_windows is either config.eval.num_test_instances for the sampled window style
        # or (T - context_length - prediction_length) // dataset.window_stride + 1 for the rolling window style
        # shape: (num_parallel_samples, num_windows*num_datasets, prediction_length, num_channels)
        predictions = np.concatenate(predictions, axis=1)
        # shape: (num_parallel_samples, num_windows*num_datasets, num_channels)
        labels = np.concatenate(labels, axis=0)
        contexts = np.concatenate(contexts, axis=0)

        # system_metrics[system] = compute_metrics(predictions, labels, include=metrics)  # type: ignore

        # shape: (num_parallel_samples, num_windows*num_datasets, prediction_length, num_channels)
        # or (num_windows*num_datasets, prediction_length, num_channels) if parallel_sample_reduction is not none
        if return_predictions:
            system_predictions[system] = parallel_sample_reduction_fn(
                predictions
            ).transpose(0, 2, 1)
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


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set floating point precision
    use_tf32 = cfg.train.tf32
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        use_tf32 = False

    # set random seed
    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

    # get test data paths
    test_data_dir = os.path.expandvars(cfg.eval.data_path)
    test_data_dict = {}
    for system_dir in Path(test_data_dir).iterdir():
        if system_dir.is_dir():
            system_name = system_dir.name
            system_files = list(system_dir.glob("*"))
            test_data_dict[system_name] = [
                FileDataset(path=Path(file_path), freq="h", one_dim_target=False)
                for file_path in system_files
                if file_path.is_file()
            ]

    log_on_main(f"Running evaluation on {list(test_data_dict.keys())}", logger)

    transforms = [
        FixedDimensionDelayEmbeddingTransform(embedding_dim=cfg.fixed_dim),
        QuadraticEmbeddingTransform(),
    ]
    test_datasets = {
        system_name: PatchTSTDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])]
            * len(test_data_dict[system_name]),
            context_length=cfg.patchtst.context_length,
            prediction_length=cfg.patchtst.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            transforms=transforms,
            mode="test",
        )
        for system_name in test_data_dict
    }

    model = PatchTST.from_pretrained(
        mode=cfg.eval.mode,
        pretrain_path=cfg.eval.checkpoint_path,
        device=cfg.eval.device,
    )
    model.eval()

    save_eval_results = partial(
        save_evaluation_results,
        metrics_save_dir=cfg.eval.metrics_save_dir,
        metrics_fname=cfg.eval.metrics_fname,
        overwrite=cfg.eval.overwrite,
        split_coords=cfg.eval.split_coords,
        verbose=cfg.eval.verbose,
    )

    if cfg.eval.mode == "predict":
        predictions, contexts, labels, metrics = evaluate_forecasting_model(
            model,
            test_datasets,
            batch_size=cfg.eval.batch_size,
            prediction_length=cfg.eval.prediction_length,
            limit_prediction_length=cfg.eval.limit_prediction_length,
            metrics=["mse", "mae", "smape", "mape", "r2_score", "spearman", "pearson"],
            return_predictions=True,
            return_contexts=True,
            return_labels=True,
            parallel_sample_reduction="mean",
            redo_normalization=True,
        )
        window_indices = None
        # # get the indices of each prediction window for each timeseries in each system
        # window_indices = rolling_prediction_window_indices(
        #     test_data_dict,
        #     cfg.eval.window_stride,
        #     cfg.patchtst.context_length,
        #     cfg.patchtst.prediction_length,
        # )

        if predictions is not None and contexts is not None:
            full_trajs = {}
            for system in predictions:
                if system not in contexts:
                    raise ValueError(f"System {system} not in contexts")
                full_trajs[system] = np.concatenate(
                    [contexts[system], predictions[system]], axis=2
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
                full_trajs[system] = np.concatenate(
                    [contexts[system], labels[system]], axis=2
                )
            save_eval_results(
                metrics,
                coords=full_trajs,
                window_indices=window_indices,
                coords_save_dir=cfg.eval.labels_save_dir,
            )

    elif cfg.eval.mode == "pretrain":
        completions, processed_past_values, masked_past_values, metrics = (
            evaluate_mlm_model(
                model,
                test_datasets,
                batch_size=cfg.eval.batch_size,
                num_systems=10,
                undo_normalization=False,
                return_completions=True,
                return_processed_past_values=True,
                return_masked_past_values=False,
            )
        )
        # TODO: plot completions, or save to npy or arrow files
        if completions is not None:
            save_eval_results(
                metrics,
                coords=completions,
                coords_save_dir=cfg.eval.completions_save_dir,
            )
        if processed_past_values is not None:
            save_eval_results(
                metrics,
                coords=processed_past_values,
                coords_save_dir=cfg.eval.patch_input_save_dir,
            )
        if masked_past_values is not None:
            save_eval_results(
                metrics,
                coords=masked_past_values,
                coords_save_dir=cfg.eval.masked_input_save_dir,
            )
    else:
        raise ValueError(f"Invalid eval mode: {cfg.eval.mode}")


if __name__ == "__main__":
    main()
