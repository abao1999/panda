"""
This script computes distributional metrics for a given model and dataset.

It computes the following metrics:
- Average Hellinger distance
- Generalized (correlation) dimension
- Maximum Lyapunov exponents
- KL divergence

See our notebook in notebooks/plot_distributional_metrics.ipynb and notebooks/plot_lyapunov_comparison.ipynb for more details on use case

TODO: Create a separate script to handle the max Lyapunov exponent computation
"""

import json
import logging
import os
import pickle
import time
from functools import partial
from multiprocessing import Pool

import dysts.flows as flows  # type: ignore
import hydra
import numpy as np
import torch
from dysts.analysis import (  # type: ignore
    gp_dim,
    max_lyapunov_exponent_rosenstein_multivariate,
)
from dysts.metrics import (  # type: ignore
    average_hellinger_distance,
    estimate_kl_divergence,
)
from tqdm import tqdm

from panda.chronos.pipeline import ChronosPipeline
from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils.eval_utils import get_eval_data_dict
from panda.utils.train_utils import log_on_main

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


def _compute_metrics_worker(
    args,
) -> tuple[str, dict[str, dict[str, float | None]]]:
    """
    Compute distributional metrics for a single system, including Hellinger distance between power spectra,
    KL divergence, correlation dimension, and maximum Lyapunov exponents
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
            - compute_dataset_stats_flag (bool): Whether to compute dataset-level stats (e.g., for full trajectory).
            - rosenstein_traj_len (int): Trajectory length parameter for Lyapunov exponent estimation.

    Returns:
        tuple:
            (system_name, {
                "prediction_horizon": {
                    "avg_hellinger_distance": float | None,
                    "gpdim_gt": float | None,
                    "gpdim_pred": float | None,
                    "max_lyap_gt": float | None,
                    "max_lyap_pred": float | None,
                },
                "pred_with_context": {
                    "avg_hellinger_distance": float | None,
                    "gpdim_gt_with_context": float | None,
                    "gpdim_pred_with_context": float | None,
                    "max_lyap_gt_with_context": float | None,
                    "max_lyap_pred_with_context": float | None,
                },
                "full_trajectory": {
                    "avg_hellinger_distance": float | None,
                    "kl_divergence": float | None,
                    "gpdim_full_traj": float | None,
                    "max_lyap_full_traj": float | None,
                },
                "prediction_time": float,
            })

    Notes:
        - All metrics may be None if computation fails for a given system.
        - All arrays are transposed to (T, d) shape before metric computation.
        - "prediction_horizon" compares predictions and ground truth over the forecast interval.
        - "pred_with_context" compares (context + predictions) to (context + ground truth).
        - "full_trajectory" compares predictions to the full system trajectory.
        - "prediction_time" is the elapsed time for generating predictions.
        - Lyapunov exponents are computed using the Rosenstein method with the provided trajectory length.
        - If compute_dataset_stats_flag is False, full-trajectory metrics for GP dimension and Lyapunov exponent are not computed (set to None).
    """
    (
        system_name,
        data,
        pred_interval,
        compute_dataset_stats_flag,
        rosenstein_traj_len,
    ) = args

    # Ensure (T, d) shape for all arrays
    context, predictions, groundtruth, full_trajectory = (
        data["context"].T,
        data["predictions"].T,
        data["groundtruth"].T,
        data["full_trajectory"].T,
    )

    predictions = predictions[:pred_interval]
    groundtruth = groundtruth[:pred_interval]

    elapsed_time = data["elapsed_time"]

    if predictions.shape != groundtruth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs groundtruth {groundtruth.shape} for {system_name}"
        )
    if context.shape[0] != 512:
        raise ValueError(f"Context length mismatch: {context.shape[0]} != 512 for {system_name}")
    if context.shape[1] != predictions.shape[1] or predictions.shape[1] != groundtruth.shape[1]:
        raise ValueError(
            f"Dimension mismatch: {context.shape[1]}, {predictions.shape[1]}, {groundtruth.shape[1]} for {system_name}"
        )

    def safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(e)
            logger.warning(f"Error computing {fn.__name__} for {system_name}")
            return None

    # (Context + GT) vs (Context + Preds)
    pred_with_context = np.concatenate([context, predictions], axis=0)
    gt_with_context = np.concatenate([context, groundtruth], axis=0)

    avg_hellinger_pred_with_context = safe_call(average_hellinger_distance, gt_with_context, pred_with_context)
    gpdim_gt_with_context = safe_call(gp_dim, gt_with_context)
    gpdim_pred_with_context = safe_call(gp_dim, pred_with_context)

    # Prediction Horizon
    avg_hellinger_pred_horizon = safe_call(average_hellinger_distance, groundtruth, predictions)
    gpdim_gt = safe_call(gp_dim, groundtruth)
    gpdim_pred = safe_call(gp_dim, predictions)

    # Prediction vs Full Trajectory
    avg_hellinger_full_traj = safe_call(average_hellinger_distance, full_trajectory, predictions)
    kl_full_traj = safe_call(estimate_kl_divergence, full_trajectory, predictions, sigma_scale=None)
    # NOTE: this is redundant computation because it should be the same for every context window pkl file,
    # but we do this for convenience when running notebooks/plot_distributional_metrics.ipynb
    if compute_dataset_stats_flag:
        gpdim_full_traj = safe_call(gp_dim, full_trajectory)
    else:
        gpdim_full_traj = None

    # Get the average integration dt for purpose of Lyapunov exponent computation
    full_traj_len = full_trajectory.shape[0]
    # we are keeping this fixed for now, but make adaptive in the future for mixed period datasets
    num_periods = 40

    system_name_without_pp = system_name.split("_pp")[0]
    is_skew = "_" in system_name_without_pp
    if is_skew:
        driver_name, response_name = system_name_without_pp.split("_")
        driver_system = getattr(flows, driver_name)()
        response_system = getattr(flows, response_name)()
        period = max(driver_system.period, response_system.period)
        # NOTE: we can use either, but it seems that for average dt, using the period is better
        # dt = min(driver_system.dt, response_system.dt)
        avg_dt = (num_periods * period) / full_traj_len

    else:
        sys = getattr(flows, system_name_without_pp)()
        avg_dt = (num_periods * sys.period) / full_traj_len

    max_lyap_gt = safe_call(
        max_lyapunov_exponent_rosenstein_multivariate,
        groundtruth,
        tau=avg_dt,
        trajectory_len=rosenstein_traj_len,
    )
    max_lyap_pred = safe_call(
        max_lyapunov_exponent_rosenstein_multivariate,
        predictions,
        tau=avg_dt,
        trajectory_len=rosenstein_traj_len,
    )
    max_lyap_gt_with_context = safe_call(
        max_lyapunov_exponent_rosenstein_multivariate,
        gt_with_context,
        tau=avg_dt,
        trajectory_len=rosenstein_traj_len,
    )
    max_lyap_pred_with_context = safe_call(
        max_lyapunov_exponent_rosenstein_multivariate,
        pred_with_context,
        tau=avg_dt,
        trajectory_len=rosenstein_traj_len,
    )
    # NOTE: this is redundant computation because it should be the same for every context window pkl file,
    # but we do this for convenience when running notebooks/plot_distributional_metrics.ipynb
    if compute_dataset_stats_flag:
        max_lyap_full_traj = safe_call(
            max_lyapunov_exponent_rosenstein_multivariate,
            full_trajectory,
            tau=avg_dt,
            trajectory_len=rosenstein_traj_len,
        )
    else:
        max_lyap_full_traj = None

    return system_name, {
        "prediction_horizon": {
            "avg_hellinger_distance": avg_hellinger_pred_horizon,
            "gpdim_gt": gpdim_gt,
            "gpdim_pred": gpdim_pred,
            "max_lyap_gt": max_lyap_gt,
            "max_lyap_pred": max_lyap_pred,
        },
        "pred_with_context": {
            "avg_hellinger_distance": avg_hellinger_pred_with_context,
            "gpdim_gt_with_context": gpdim_gt_with_context,
            "gpdim_pred_with_context": gpdim_pred_with_context,
            "max_lyap_gt_with_context": max_lyap_gt_with_context,
            "max_lyap_pred_with_context": max_lyap_pred_with_context,
        },
        "full_trajectory": {
            "avg_hellinger_distance": avg_hellinger_full_traj,
            "kl_divergence": kl_full_traj,
            "gpdim_full_traj": gpdim_full_traj,
            "max_lyap_full_traj": max_lyap_full_traj,
        },
        "prediction_time": elapsed_time,
    }


def get_distributional_metrics(
    forecast_dict: dict[str, dict[str, np.ndarray]],
    n_jobs: int | None = None,
    compute_dataset_stats: bool = False,
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
        n_jobs (int | None, optional):
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
    # pred_intervals = [128, 192, 256, 320, 384, 448, 512]
    pred_intervals = [128, 256, 512]
    # NOTE: prediction length 128 is too short to use traj_len=64 for the rosenstein estimator implementation we consider.
    rosenstein_traj_lens = [16, 64, 64]

    compute_dataset_stats_flag = compute_dataset_stats

    for i, pred_interval in enumerate(pred_intervals):
        # if i == 0 and compute_dataset_stats
        # Prepare arguments for parallel processing
        current_rosenstein_traj_len = rosenstein_traj_lens[i]
        worker_args = [
            (
                system_name,
                data,
                pred_interval,
                compute_dataset_stats_flag,
                current_rosenstein_traj_len,
            )
            for system_name, data in forecast_dict.items()
        ]
        print(f"pred_interval: {pred_interval}")
        print(f"compute_dataset_stats_flag: {compute_dataset_stats_flag}")
        print(f"rosenstein_traj_len: {current_rosenstein_traj_len}")
        # Use multiprocessing to compute dimensions in parallel
        with Pool(processes=n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(_compute_metrics_worker, worker_args),
                    total=len(worker_args),
                    desc="Computing distributional metrics",
                )
            )
        results_all_pred_intervals[pred_interval] = results

    return results_all_pred_intervals


def get_model_prediction(
    model,
    context: np.ndarray,
    prediction_length: int,
    is_chronos: bool = False,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """
    Generate model predictions for a given context and prediction length.

    Args:
        model: The model to use for prediction.
        context (np.ndarray): The input context array. Shape should be (dim, timesteps).
        prediction_length (int): The number of timesteps to predict.
        is_chronos (bool, optional): If True, use Chronos-specific input/output conventions. Default is False.
        **kwargs: Additional keyword arguments to pass to the model's predict method.

    Returns:
        tuple[np.ndarray, float]:
            - pred (np.ndarray): The predicted values, with shape (dim, timesteps).
            - elapsed_time (float): The time taken for prediction in seconds.
    """
    context_tensor = (
        torch.from_numpy(context.T if not is_chronos else context).float()
        # .to(model.device)
    )
    if not is_chronos:
        context_tensor = context_tensor[None, ...]

    start_time = time.time()
    pred = model.predict(context_tensor, prediction_length, **kwargs).squeeze().cpu().numpy()
    elapsed_time = time.time() - start_time

    if is_chronos:
        if not kwargs.get("deterministic", False):
            pred = np.median(pred, axis=1)
    else:
        pred = pred.T

    return pred, elapsed_time


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    metrics_save_dir = cfg.eval.metrics_save_dir
    os.makedirs(metrics_save_dir, exist_ok=True)

    forecasts_dict_path = os.path.join(metrics_save_dir, f"{cfg.eval.metrics_fname}_forecasts.pkl")

    if not cfg.eval.reload_saved_forecasts:
        checkpoint_path = cfg.eval.checkpoint_path
        if not cfg.eval.chronos.zero_shot:
            log(f"Using checkpoint: {checkpoint_path}")
        else:
            log(f"Using Chronos Zeroshot: {cfg.chronos.model_id}")

        # torch_dtype = getattr(torch, cfg.eval.torch_dtype)
        # assert isinstance(torch_dtype, torch.dtype)

        model_kwargs = {
            "panda": {
                "limit_prediction_length": False,
                "sliding_context": True,
                "is_chronos": False,
                "verbose": False,
            },
            "chronos": {
                "is_chronos": True,
                "limit_prediction_length": False,
                "num_samples": cfg.eval.num_samples,
                "deterministic": True if cfg.eval.num_samples == 1 else False,
                "verbose": False,
            },
        }[cfg.eval.model_type]

        if cfg.eval.model_type == "panda":
            model_pipeline = PatchTSTPipeline.from_pretrained(
                mode=cfg.eval.mode,
                pretrain_path=checkpoint_path,
                device_map=cfg.eval.device,
                # torch_dtype=torch_dtype,
            )
        elif cfg.eval.model_type == "chronos":
            model_pipeline = ChronosPipeline.from_pretrained(
                cfg.chronos.model_id if cfg.eval.chronos.zero_shot else checkpoint_path,
                device_map=cfg.eval.device,
                torch_dtype=torch.float32,
            )
        else:
            raise ValueError(f"Invalid model type: {cfg.eval.model_type}")

        prediction_length = cfg.eval.prediction_length
        context_length = cfg.eval.context_length
        window_start_time = cfg.eval.window_start_time
        window_end_time = window_start_time + context_length

        log(f"Using context length: {context_length} and prediction length: {prediction_length}")

        log(f"Saving forecasts to {forecasts_dict_path}")
        forecast_dict = {}
        for subdir_name, datasets in tqdm(
            list(test_data_dict.items())[: cfg.eval.num_subdirs],
            desc="Generating forecasts for subdirectories",
        ):
            log(f"Processing {len(datasets)} datasets in {subdir_name}")
            for file_dataset in datasets[: cfg.eval.num_samples_per_subdir]:
                filepath = file_dataset.iterable.path  # type: ignore
                sample_idx = int(os.path.basename(filepath).split("_")[0])
                system_name = f"{subdir_name}_pp{sample_idx}"
                coords, _ = zip(*[(coord["target"], coord["start"]) for coord in file_dataset])
                coordinates = np.stack(coords)
                if coordinates.ndim > 2:  # if not one_dim_target:
                    coordinates = coordinates.squeeze()

                context = coordinates[:, window_start_time:window_end_time]
                groundtruth = coordinates[:, window_end_time : window_end_time + prediction_length]

                preds, elapsed_time = get_model_prediction(
                    model_pipeline,
                    context=context,
                    prediction_length=prediction_length,
                    **model_kwargs,
                )
                forecast_dict[system_name] = {
                    "context": context,
                    "predictions": preds,
                    "groundtruth": groundtruth,
                    "full_trajectory": coordinates,
                    "elapsed_time": elapsed_time,
                }

        if cfg.eval.save_forecasts:
            log(f"Saving forecasts to {metrics_save_dir}")
            if not cfg.eval.save_full_trajectory:
                # cut out "full_trajectory"
                forecast_dict = {k: v for k, v in forecast_dict.items() if k != "full_trajectory"}
            with open(forecasts_dict_path, "wb") as f:
                pickle.dump(forecast_dict, f)

    else:
        log(f"Reloading forecasts from {forecasts_dict_path}")
        with open(forecasts_dict_path, "rb") as f:
            forecast_dict = pickle.load(f)

    distributional_metrics = get_distributional_metrics(
        forecast_dict,
        n_jobs=cfg.eval.num_processes,
        compute_dataset_stats=cfg.eval.compute_dataset_stats,
    )

    metrics_fname_suffix = f"_{cfg.eval.metrics_fname_suffix}" if cfg.eval.metrics_fname_suffix else ""
    metrics_fname = (
        f"{cfg.eval.metrics_fname}{metrics_fname_suffix}" if cfg.eval.reload_saved_forecasts else cfg.eval.metrics_fname
    )
    metrics_path = os.path.join(metrics_save_dir, f"{metrics_fname}.json")
    log(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(distributional_metrics, f, indent=4)


if __name__ == "__main__":
    main()
