"""
Batched distributional metrics computation using Panda's evaluation utilities.

This script mirrors the model-aware dataset construction and batched prediction
flow used by the main evaluate scripts, and then computes the same
distributional metrics produced by compute_distributional_metrics.py.

Notes:
- Uses UnivariateTimeSeriesDataset for Chronos models and MultivariateTimeSeriesDataset
  for PatchTST ("panda") models.
- Runs batched evaluation with metric_names=None (no pointwise metrics computed here).
- Builds a forecast_dict compatible with get_distributional_metrics and computes
  distributional metrics over the returned batched contexts/predictions/labels.
- To associate a full trajectory per forecast row, we enforce a single eval window
  per underlying file by using window_style="single" and num_test_instances=1.
  This guarantees a 1:1 mapping between each forecast row and its source file.
"""

import gc
import json
import logging
import os
import pickle
from functools import partial
from multiprocessing import Pool

import hydra
import numpy as np
import torch
from dysts.metrics import (  # type: ignore
    average_hellinger_distance,
    estimate_kl_divergence,
)
from gluonts.transform import LastValueImputation
from tqdm import tqdm
from transformers import set_seed

from panda.baselines.fm_baselines import DynaMixPipeline
from panda.chronos.pipeline import ChronosPipeline
from panda.dataset import MultivariateTimeSeriesDataset, UnivariateTimeSeriesDataset
from panda.evaluation import (
    evaluate_multivariate_forecasting_model,
    evaluate_univariate_forecasting_model,
)
from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils.data_utils import get_dim_from_dataset
from panda.utils.eval_utils import get_eval_data_dict
from panda.utils.train_utils import log_on_main

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)

UNIVARIATE_MODELS = ["chronos"]
MULTIVARIATE_MODELS = ["panda", "dynamix"]


def _compute_metrics_worker(
    args,
) -> tuple[str, dict[str, dict[str, float | None]]]:
    """
    Compute distributional metrics for a single system, including Hellinger distance between power spectra,
    KL divergence, correlation dimension
    for various combinations of context, predictions, ground truth, and full trajectory.

    Args:
        args: Tuple containing:
            - system_name (str): Name of the dynamical system.
            - data (dict): Dictionary with the following keys:
                - "context": np.ndarray, shape (d, T)
                - "predictions": np.ndarray, shape (d, T)
                - "groundtruth": np.ndarray, shape (d, T)
                - "full_trajectory": np.ndarray, shape (d, T_full)
                - "elapsed_time": float
            - pred_interval (int): Number of prediction steps to consider.

    Returns:
        tuple:
            (system_name, {
                "prediction_horizon": {
                    "avg_hellinger_distance": float | None,
                },
                "full_trajectory": {
                    "avg_hellinger_distance": float | None,
                    "kl_divergence": float | None,
                },
                "prediction_time": float,
            })

    Notes:
        - All metrics may be None if computation fails for a given system.
        - All arrays are transposed to (T, d) shape before metric computation.
        - "prediction_horizon" compares predictions and ground truth over the forecast interval.
        - "full_trajectory" compares predictions to the full system trajectory.
        - "prediction_time" is the elapsed time for generating predictions.
    """
    (
        system_name,
        data,
        pred_interval,
    ) = args

    # Ensure (T, d) shape for all arrays - only load what we need
    predictions = data["predictions"].T[:pred_interval]
    groundtruth = data["groundtruth"].T[:pred_interval]
    full_trajectory = data["full_trajectory"].T

    elapsed_time = data["elapsed_time"]

    if predictions.shape != groundtruth.shape:
        raise ValueError(
            f"Shape mismatch for {system_name}: predictions {predictions.shape} vs groundtruth {groundtruth.shape}"
        )
    if predictions.shape[1] != groundtruth.shape[1]:
        raise ValueError(
            f"Dimension mismatch for {system_name}: pred={predictions.shape[1]}, gt={groundtruth.shape[1]}"
        )

    def safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(e)
            logger.warning(f"Error computing {fn.__name__} for {system_name}")
            return None

    # Prediction Horizon
    avg_hellinger_pred_horizon = safe_call(average_hellinger_distance, groundtruth, predictions)
    kl_pred_horizon = safe_call(estimate_kl_divergence, groundtruth, predictions, n_samples=1000, sigma_scale=None)

    # Prediction vs Full Trajectory
    avg_hellinger_full_traj = safe_call(average_hellinger_distance, full_trajectory, predictions)
    kl_full_traj = safe_call(estimate_kl_divergence, full_trajectory, predictions, n_samples=1000, sigma_scale=None)

    result = {
        "prediction_horizon": {
            "avg_hellinger_distance": avg_hellinger_pred_horizon,
            "kl_divergence": kl_pred_horizon,
        },
        "full_trajectory": {
            "avg_hellinger_distance": avg_hellinger_full_traj,
            "kl_divergence": kl_full_traj,
        },
        "prediction_time": elapsed_time,
    }

    # Clean up large arrays before returning to free memory
    del predictions, groundtruth, full_trajectory
    gc.collect()

    return system_name, result


def get_distributional_metrics(
    forecast_dict: dict[str, dict[str, np.ndarray | float]],
    pred_intervals: list[int],
    use_multiprocessing: bool = True,
    n_proc: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute distributional metrics (average Hellinger distance and KL divergence)
    for multiple systems in parallel.

    This function processes a dictionary of forecast data for multiple systems,
    computes distributional metrics comparing model predictions to ground truth
    and full trajectories, and returns the results in a nested dictionary.

    Args:
        forecast_dict (dict[str, dict[str, np.ndarray]]):
            A dictionary mapping system names to dictionaries containing the following keys:
                - "context": np.ndarray, context data of shape (d, T)
                - "predictions": np.ndarray, model predictions of shape (d, T)
                - "groundtruth": np.ndarray, ground truth data of shape (d, T)
                - "full_trajectory": np.ndarray, full trajectory data of shape (d, T)
            Additional keys may be present but are ignored by this function.
        n_proc (int | None, optional):
            Number of worker processes to use for parallel computation.
            If None, uses all available CPU cores.

    Returns:
        dict[str, dict[str, dict[str, float]]]:
            A dictionary mapping each system name to a dictionary with keys
            "prediction_horizon" and "full_trajectory". Each of these contains
            a dictionary with the computed metrics:
                - "avg_hellinger_distance": float
                - "kl_divergence": float

    Raises:
        ValueError: If any required key is missing from a system's data.
    """
    # Validate all data upfront
    required_keys = ["context", "predictions", "groundtruth", "full_trajectory"]
    for system_name, data in forecast_dict.items():
        if not all(key in data for key in required_keys):
            raise ValueError(f"Missing required data for {system_name}: {required_keys}")

    results_all_pred_intervals = {}

    for pred_interval in pred_intervals:
        worker_args = [
            (
                system_name,
                data,
                pred_interval,
            )
            for system_name, data in forecast_dict.items()
        ]
        print(f"pred_interval: {pred_interval}")

        if use_multiprocessing:
            # Use multiprocessing to compute dimensions in parallel
            # maxtasksperchild prevents memory buildup in worker processes
            with Pool(n_proc, maxtasksperchild=10) as pool:
                results = []
                # Process results incrementally rather than accumulating all at once
                for result in tqdm(
                    pool.imap_unordered(_compute_metrics_worker, worker_args, chunksize=1),
                    total=len(worker_args),
                    desc="Computing distributional metrics",
                ):
                    results.append(result)

            # Explicitly delete worker_args to free memory
            del worker_args
            gc.collect()
        else:
            print("Computing distributional metrics sequentially")
            # Sequential (non-multiprocessed) version
            results = []
            for args in tqdm(worker_args, desc="Computing distributional metrics"):
                result = _compute_metrics_worker(args)
                results.append(result)
                # Free memory after each computation
                del args

            del worker_args
            gc.collect()

        results_all_pred_intervals[pred_interval] = results

    return results_all_pred_intervals


def _load_full_trajectories(file_datasets: list) -> np.ndarray:
    """Load full trajectory (dim, T) arrays for each FileDataset.

    The FileDataset yields entries per-dimension; we stack them along dim.
    """
    full_trajs: list[np.ndarray] = []
    for file_dataset in file_datasets:
        coords, _ = zip(*[(entry["target"], entry["start"]) for entry in file_dataset])
        coordinates = np.stack(coords)
        if coordinates.ndim > 2:
            coordinates = coordinates.squeeze()
        full_trajs.append(coordinates)
    return np.array(full_trajs)  # (num datasets, dim, traj len)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # Load datasets grouped by system name
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
        one_dim_target=False,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    # Metrics io config
    metrics_save_dir = cfg.eval.metrics_save_dir
    os.makedirs(metrics_save_dir, exist_ok=True)
    forecasts_dict_path = os.path.join(metrics_save_dir, f"{cfg.eval.metrics_fname}_forecasts.pkl")

    # If reloading forecasts, skip prediction
    if cfg.eval.reload_saved_forecasts:
        log(f"Reloading forecasts from {forecasts_dict_path}")
        with open(forecasts_dict_path, "rb") as f:
            forecast_dict = pickle.load(f)
    else:
        prediction_length = cfg.eval.prediction_length
        context_length = None  # to be set below

        log("Initializing model/pipeline...")
        # Initialize model/pipeline
        if cfg.eval.model_type == "panda":
            log(f"Using PatchTST checkpoint: {cfg.eval.checkpoint_path}")
            pipeline = PatchTSTPipeline.from_pretrained(
                mode="predict",
                pretrain_path=cfg.eval.checkpoint_path,
                device_map=cfg.eval.device,
                torch_dtype=getattr(torch, cfg.eval.torch_dtype, torch.float32),
            )
            pipeline.model.eval()
            model_config = dict(vars(pipeline.model.config))
            context_length = model_config["context_length"]

        elif cfg.eval.model_type == "dynamix":
            torch_dtype = getattr(torch, cfg.dynamix.torch_dtype)
            assert isinstance(torch_dtype, torch.dtype)
            log(f"Using Dynamix model (cfg.dynamix.model_name): {cfg.dynamix.model_name}")
            pipeline = DynaMixPipeline(
                model_name=cfg.dynamix.model_name,
                device=cfg.dynamix.device,
                torch_dtype=torch_dtype,
                preprocessing_method=cfg.dynamix.preprocessing_method,
                standardize=cfg.dynamix.standardize,
                fit_nonstationary=cfg.dynamix.fit_nonstationary,
            )

            train_config = dict(cfg.train)
            rseed = train_config.get("seed", cfg.train.seed)
            log(f"Using SEED: {rseed}")
            set_seed(seed=rseed)

            context_length = cfg.chronos.context_length

        elif cfg.eval.model_type == "chronos":
            log(f"Using Chronos checkpoint: {cfg.eval.checkpoint_path}")
            pipeline = ChronosPipeline.from_pretrained(
                cfg.eval.checkpoint_path,
                device_map=cfg.eval.device,
                torch_dtype=getattr(torch, cfg.eval.torch_dtype, torch.float32),
            )
            pipeline.model.eval()
            model_config = dict(vars(pipeline.model.config))
            context_length = model_config["context_length"]

        else:
            raise NotImplementedError(
                f"Batched distributional metrics only supports model_type in {{'panda','chronos'}}, got {cfg.eval.model_type}"
            )

        log("Loading datasets...")
        log("Context length: {context_length}")
        if cfg.eval.model_type in MULTIVARIATE_MODELS:
            datasets = {
                system_name: MultivariateTimeSeriesDataset(
                    datasets=file_datasets,
                    probabilities=[1.0 / len(file_datasets)] * len(file_datasets),
                    context_length=context_length,
                    prediction_length=prediction_length,
                    window_style="single",
                    window_start=cfg.eval.window_start,
                    model_type="predict",
                    mode="test",
                )
                for system_name, file_datasets in test_data_dict.items()
            }
        elif cfg.eval.model_type in UNIVARIATE_MODELS:
            datasets = {
                system_name: UnivariateTimeSeriesDataset(
                    datasets=file_datasets,
                    probabilities=[1.0 / len(file_datasets)] * len(file_datasets),
                    tokenizer=pipeline.tokenizer,  # type: ignore[attr-defined]
                    context_length=context_length,
                    prediction_length=prediction_length,
                    min_past=cfg.min_past,
                    window_style="single",
                    window_start=cfg.eval.window_start,
                    model_type=cfg.chronos.model_type,
                    imputation_method=LastValueImputation() if cfg.chronos.model_type == "causal" else None,
                    mode="test",
                )
                for system_name, file_datasets in test_data_dict.items()
            }

        system_dims = {
            system_name: get_dim_from_dataset(test_data_dict[system_name][0]) for system_name in test_data_dict
        }

        log(f"context_length: {context_length}")
        log(f"prediction_length: {prediction_length}")

        parallel_sample_reduction_fn = {
            "mean": lambda x: np.mean(x, axis=0),
            "median": lambda x: np.median(x, axis=0),
        }[cfg.eval.parallel_sample_reduction]

        # Run batched prediction
        if cfg.eval.model_type in MULTIVARIATE_MODELS:
            preds, ctxs, lbls, _ = evaluate_multivariate_forecasting_model(
                pipeline,
                datasets,  # type: ignore
                batch_size=cfg.eval.batch_size,
                prediction_length=prediction_length,
                metric_names=None,
                system_dims=system_dims,
                return_predictions=True,
                return_contexts=True,
                return_labels=True,
                parallel_sample_reduction_fn=parallel_sample_reduction_fn,
                redo_normalization=True,
                prediction_kwargs=dict(
                    sliding_context=False,
                    limit_prediction_length=False,
                    verbose=cfg.eval.verbose,
                ),
                num_workers=cfg.eval.dataloader_num_workers,
            )
        else:
            prediction_kwargs = {
                "limit_prediction_length": False,
                "deterministic": True if cfg.eval.chronos.deterministic else False,
                "verbose": cfg.eval.verbose,
                "top_k": cfg.chronos.top_k,
                "top_p": cfg.chronos.top_p,
                "temperature": cfg.chronos.temperature,
                "num_samples": 1 if cfg.eval.chronos.deterministic else cfg.eval.num_samples,
            }
            preds, ctxs, lbls, _ = evaluate_univariate_forecasting_model(
                pipeline,
                datasets,  # type: ignore
                batch_size=cfg.eval.batch_size,
                prediction_length=prediction_length,
                metric_names=None,
                system_dims=system_dims,
                return_predictions=True,
                return_contexts=True,
                return_labels=True,
                parallel_sample_reduction_fn=parallel_sample_reduction_fn,
                prediction_kwargs=prediction_kwargs,
                num_workers=cfg.eval.dataloader_num_workers,
            )

        # Assemble forecast_dict with full trajectory mapping
        log("Assembling forecast dictionary for distributional metrics")
        forecast_dict: dict[str, dict[str, np.ndarray | float]] = {}

        for system_name, file_datasets in tqdm(test_data_dict.items(), desc="Preparing forecast_dict"):
            full_trajs = _load_full_trajectories(file_datasets)

            system_preds = preds[system_name] if preds is not None else None
            system_ctxs = ctxs[system_name] if ctxs is not None else None
            system_lbls = lbls[system_name] if lbls is not None else None

            if system_preds is None or system_ctxs is None or system_lbls is None:
                raise RuntimeError(f"Missing predictions/contexts/labels for system {system_name}")

            # Expect one row per file because window_style='single'
            num_rows = system_preds.shape[0]
            if num_rows != len(full_trajs):
                raise RuntimeError(
                    f"Row/file count mismatch for {system_name}: rows={num_rows}, files={len(full_trajs)}."
                )

            for i in range(num_rows):
                key = f"{system_name}_pp{i}"
                context_i = system_ctxs[i]  # (dim, context_length)
                preds_i = system_preds[i]  # (dim, prediction_length)
                labels_i = system_lbls[i]  # (dim, prediction_length)
                full_traj_i = full_trajs[i]  # (dim, T_full)

                forecast_dict[key] = {
                    "context": context_i,
                    "predictions": preds_i,
                    "groundtruth": labels_i,
                    "full_trajectory": full_traj_i,
                    "elapsed_time": 0.0,
                }

        if cfg.eval.save_forecasts:
            log(f"Saving forecasts to {metrics_save_dir}")
            save_dict = (
                forecast_dict
                if cfg.eval.save_full_trajectory
                else {k: {kk: vv for kk, vv in v.items() if kk != "full_trajectory"} for k, v in forecast_dict.items()}
            )
            with open(forecasts_dict_path, "wb") as f:
                pickle.dump(save_dict, f)

    # Compute the distributional metrics
    if cfg.eval.compute_distributional_metrics:
        log("Computing distributional metrics")
        pred_intervals = [1024]
        distributional_metrics = get_distributional_metrics(
            forecast_dict,
            pred_intervals,
            use_multiprocessing=cfg.eval.use_multiprocessing,
            n_proc=cfg.eval.num_processes,
        )

        metrics_fname_suffix = f"_{cfg.eval.metrics_fname_suffix}" if cfg.eval.metrics_fname_suffix else ""
        metrics_fname = (
            f"{cfg.eval.metrics_fname}{metrics_fname_suffix}"
            if cfg.eval.reload_saved_forecasts
            else cfg.eval.metrics_fname
        )
        metrics_path = os.path.join(metrics_save_dir, f"{metrics_fname}.json")
        log(f"Saving metrics to {metrics_path}")
        with open(metrics_path, "w") as f:
            json.dump(distributional_metrics, f, indent=4)
    else:
        # We do this so we can parallelize the model inference and not worry about cpu usage bottleneck (computing these metrics is very cpu intensive)
        log(f"Skipping distributional metrics computation, only saved forecasts, to {forecasts_dict_path}")


if __name__ == "__main__":
    main()
