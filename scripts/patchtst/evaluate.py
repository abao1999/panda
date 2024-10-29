import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import transformers
from dysts.metrics import compute_metrics  # type: ignore
from gluonts.dataset.common import FileDataset
from gluonts.itertools import batcher
from omegaconf import DictConfig
from tqdm.auto import tqdm

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTST
from dystformer.utils import log_on_main, process_trajs

logger = logging.getLogger(__name__)


def evaluate_mlm_model(
    model: PatchTST,
    systems: Dict[str, PatchTSTDataset],
    batch_size: int,
    metrics: Optional[List[str]] = None,
    return_completions: bool = False,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    """
    past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
    Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
    in `[0, 1]`:
    """
    assert model.mode == "pretrain", "Model must be in pretrain mode"
    system_completions = {}
    system_metrics = {system: defaultdict(float) for system in systems}

    for system in tqdm(systems, desc="Evaluating MLM pretrain model"):
        dataset = systems[system]  # IterableDataset
        print(f"Evaluating {system}")
        all_completions = []
        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            past_values = [data["past_values"] for data in batch]
            past_batch = torch.from_numpy(np.stack(past_values)).to(model.device)

            # TODO: PatchTSTModel self.masking and mask in PatchTSTModel forward returned in PatchTSTModelOutput
            past_masked = None

            # see patchtstforpretraining
            completions = (
                model(past_batch, past_observed_mask=None).transpose(1, 0).cpu().numpy()
            )

            eval_metrics = compute_metrics(completions, past_masked, include=metrics)

            # compute running average of metrics over batches
            for metric, value in eval_metrics.items():
                system_metrics[system][metric] += (
                    value - system_metrics[system][metric]
                ) / (i + 1)

            if return_completions:
                all_completions.append(completions)

        if return_completions:
            full_completion = np.concatenate(all_completions, axis=1)
            system_completions[system] = full_completion

    # convert defaultdicts to regular dicts
    system_metrics = {
        system: dict(metrics) for system, metrics in system_metrics.items()
    }

    return system_completions if return_completions else None, system_metrics


def evaluate_forecasting_model(
    model: PatchTST,
    systems: Dict[str, PatchTSTDataset],
    batch_size: int,
    prediction_length: int,
    limit_prediction_length: bool = False,
    metrics: Optional[List[str]] = None,
    parallel_sample_reduction: str = "none",
    return_predictions: bool = False,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
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
    Returns:
        A tuple containing:
        - system_predictions: A dictionary mapping system names to their predictions.
            Only returned if `return_predictions` is True.
        - system_metrics: A nested dictionary containing computed metrics for each system.
    """
    assert model.mode == "predict", "Model must be in predict mode"
    system_predictions = {}
    system_metrics = {system: {} for system in systems}

    parallel_sample_reduction_fn = {
        "mean": lambda x: np.mean(x, axis=0),
        "median": lambda x: np.median(x, axis=0),
    }.get(parallel_sample_reduction, lambda x: x)

    for system in tqdm(systems, desc="Evaluating model"):
        dataset = systems[system]
        predictions, labels = [], []
        for batch in batcher(dataset, batch_size=batch_size):
            past_values, future_values = zip(
                *[(data["past_values"], data["future_values"]) for data in batch]
            )
            past_batch = torch.stack(past_values, dim=0).to(model.device)
            preds = model.predict(
                past_batch,
                prediction_length=prediction_length,
                limit_prediction_length=limit_prediction_length,
            ).transpose(1, 0)

            future_batch = torch.stack(future_values, dim=0)

            labels.append(future_batch)
            predictions.append(preds)

        # num_windows is either config.eval.num_test_instances for the sampled window style
        # or (T - context_length - prediction_length) // dataset.window_stride + 1 for the rolling window style
        # shape: (num_parallel_samples, num_windows*num_datasets, prediction_length, num_channels)
        predictions = torch.cat(predictions, dim=1).cpu().numpy()
        # shape: (num_parallel_samples, num_windows*num_datasets, num_channels)
        labels = torch.cat(labels, dim=0).cpu().numpy()

        eval_metrics = compute_metrics(predictions, labels, include=metrics)
        system_metrics[system] = eval_metrics

        # shape: (num_parallel_samples, num_windows*num_datasets, prediction_length, num_channels)
        # or (num_windows*num_datasets, prediction_length, num_channels) if parallel_sample_reduction is not none
        if return_predictions:
            system_predictions[system] = parallel_sample_reduction_fn(predictions)

    return system_predictions if return_predictions else None, system_metrics


def save_predictions(
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    eval_cfg: DictConfig,
):
    """
    Save prediction metrics and optionally forecast trajectories.

    Args:
        predictions: Dictionary mapping system names to prediction numpy arrays.
        metrics: Nested dictionary containing computed metrics for each system.
        eval_cfg: Configuration object containing evaluation settings.

    This function performs two main tasks:
    1. Saves evaluation metrics to a CSV file, appending to existing file if present.
    2. If specified in eval_cfg, saves forecast trajectories as arrow files.
    """
    result_rows = []
    result_rows.extend(
        {
            "system": system,
            **metrics[system],
        }
        for system in metrics
    )
    results_df = pd.DataFrame(result_rows)

    metrics_fname = eval_cfg.output_fname or f"{eval_cfg.split}_metrics.csv"
    metrics_save_path = os.path.join(eval_cfg.output_dir, metrics_fname)
    print("Saving metrics to: ", metrics_save_path)
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

    if os.path.isfile(metrics_save_path) and not eval_cfg.overwrite:
        existing_df = pd.read_csv(metrics_save_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(metrics_save_path, index=False)

    # save predictions, which is a dictionary mapping system names to prediction numpy arrays, to arrow files
    if eval_cfg.forecast_save_dir is not None and predictions is not None:
        print(f"forecast dysts: {predictions.keys()}")
        for system in predictions:
            print(f"{system}: {predictions[system].shape}")

        os.makedirs(eval_cfg.forecast_save_dir, exist_ok=True)
        print(
            f"Saving all valid sampled trajectories from {len(predictions)} systems to arrow files within {eval_cfg.forecast_save_dir}"
        )
        process_trajs(
            eval_cfg.forecast_save_dir,
            predictions,
            split_coords=eval_cfg.split_coords,
            verbose=eval_cfg.verbose,
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

    for system_name in test_data_dict:
        print(f"{system_name}: {len(test_data_dict[system_name])}")

    log_on_main(f"Running evaluation on {list(test_data_dict.keys())}", logger)

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
            mode="test",
        )
        for system_name in test_data_dict
    }

    model = PatchTST(
        cfg.patchtst,
        mode=cfg.eval.mode,
        pretrained_encoder_path=cfg.patchtst.pretrained_encoder_path,
        device=cfg.eval.device,
    )
    model.eval()

    predictions, metrics = evaluate_forecasting_model(
        model,
        test_datasets,
        batch_size=cfg.eval.batch_size,
        prediction_length=cfg.patchtst.prediction_length,
        limit_prediction_length=cfg.eval.limit_prediction_length,
        metrics=[
            "mse",
            "mae",
            "smape",
            "mape",
            "r2_score",
            "spearman",
            "pearson",
        ],
        return_predictions=True,
    )

    if predictions is not None:
        save_predictions(predictions, metrics, cfg.eval)


if __name__ == "__main__":
    main()
