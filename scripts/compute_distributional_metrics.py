import json
import logging
import os
import pickle
import time
from functools import partial
from multiprocessing import Pool

import hydra
import numpy as np
import torch
from dysts.analysis import (  # type: ignore
    corr_gpdim,
    gp_dim,
    max_lyapunov_exponent_rosenstein,
)
from dysts.metrics import (  # type: ignore
    average_hellinger_distance,
    estimate_kl_divergence,
)
from tqdm import tqdm

from panda.chronos.pipeline import ChronosPipeline
from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils import (
    get_eval_data_dict,
    log_on_main,
)

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


def _compute_dataset_metrics_worker(
    args,
) -> tuple[str, dict[str, float | None]]:
    """
    Compute distributional metrics (average Hellinger distance and KL divergence)
    for a single system.

    Args:
        args: Tuple of (system_name, data_dict) where data_dict contains
              "context", "predictions", "groundtruth", "full_trajectory" as np.ndarrays.

    Returns:
        (system_name, {
            "max_lyap_rosenstein": float,
        })
    """
    dyst_name, data, _ = args

    # Ensure (T, d) shape for all arrays
    _, _, _, full_trajectory = (
        data["context"].T,
        data["predictions"].T,
        data["groundtruth"].T,
        data["full_trajectory"].T,
    )

    def safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.warning(f"Error computing {fn.__name__} for {dyst_name}")
            return None

    max_lyap_full_traj = safe_call(max_lyapunov_exponent_rosenstein, full_trajectory)

    return dyst_name, {
        "max_lyap_rosenstein": max_lyap_full_traj,
    }


def _compute_more_metrics_worker(
    args,
) -> tuple[str, dict[str, dict[str, float | None]]]:
    """
    Compute distributional metrics (average Hellinger distance and KL divergence)
    for a single system.

    Args:
        args: Tuple of (system_name, data_dict) where data_dict contains
              "context", "predictions", "groundtruth", "full_trajectory" as np.ndarrays.

    Returns:
        (system_name, {
            "pred_with_context": {
                "avg_hellinger_distance": float,
                "kl_divergence": float,
                "corr_gpdim_pred_with_context": float,
                "gpdim_gt_with_context": float,
                "gpdim_pred_with_context": float,
            },
            "max_lyap_rosenstein": {
                "max_lyap_context": float,
                "max_lyap_full_traj": float,
            }
        })
    """
    dyst_name, data, pred_interval = args

    # Ensure (T, d) shape for all arrays
    context, predictions, groundtruth, _ = (
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
            f"Shape mismatch: predictions {predictions.shape} vs groundtruth {groundtruth.shape} for {dyst_name}"
        )
    if context.shape[0] != 512:
        raise ValueError(
            f"Context length mismatch: {context.shape[0]} != 512 for {dyst_name}"
        )
    if (
        context.shape[1] != predictions.shape[1]
        or predictions.shape[1] != groundtruth.shape[1]
    ):
        raise ValueError(
            f"Dimension mismatch: {context.shape[1]}, {predictions.shape[1]}, {groundtruth.shape[1]} for {dyst_name}"
        )

    def safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.warning(f"Error computing {fn.__name__} for {dyst_name}")
            return None

    pred_with_context = np.concatenate([context, predictions], axis=0)
    gt_with_context = np.concatenate([context, groundtruth], axis=0)

    avg_hellinger_pred_with_context = safe_call(
        average_hellinger_distance, gt_with_context, pred_with_context
    )
    kl_pred_with_context = safe_call(
        estimate_kl_divergence, gt_with_context, pred_with_context
    )

    corr_gpdim_result = safe_call(corr_gpdim, gt_with_context, pred_with_context)
    if corr_gpdim_result is not None:
        corr_gpdim_pred_with_context, gpdim_gt_with_context, gpdim_pred_with_context = (
            corr_gpdim_result
        )
    else:
        gpdim_gt_with_context = safe_call(gp_dim, gt_with_context)
        gpdim_pred_with_context = safe_call(gp_dim, pred_with_context)

    return dyst_name, {
        "pred_with_context": {
            "avg_hellinger_distance": avg_hellinger_pred_with_context,
            "kl_divergence": kl_pred_with_context,
            "corr_gpdim_pred_with_context": corr_gpdim_pred_with_context,
            "gpdim_gt_with_context": gpdim_gt_with_context,
            "gpdim_pred_with_context": gpdim_pred_with_context,
        },
        "prediction_time": elapsed_time,
    }


def _compute_metrics_worker(args) -> tuple[str, dict[str, dict[str, float | None]]]:
    """
    Compute distributional metrics (average Hellinger distance and KL divergence)
    for a single system.

    Args:
        args: Tuple of (system_name, data_dict) where data_dict contains
              "context", "predictions", "groundtruth", "full_trajectory" as np.ndarrays.

    Returns:
        (system_name, {
            "prediction_horizon": {"avg_hellinger_distance": float, "kl_divergence": float},
            "full_trajectory": {"avg_hellinger_distance": float, "kl_divergence": float}
        })
    """
    dyst_name, data, pred_interval = args

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
            f"Shape mismatch: predictions {predictions.shape} vs groundtruth {groundtruth.shape} for {dyst_name}"
        )
    if context.shape[0] != 512:
        raise ValueError(
            f"Context length mismatch: {context.shape[0]} != 512 for {dyst_name}"
        )
    if (
        context.shape[1] != predictions.shape[1]
        or predictions.shape[1] != groundtruth.shape[1]
    ):
        raise ValueError(
            f"Dimension mismatch: {context.shape[1]}, {predictions.shape[1]}, {groundtruth.shape[1]} for {dyst_name}"
        )

    def safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.warning(f"Error computing {fn.__name__} for {dyst_name}")
            return None

    max_lyap_gt = safe_call(max_lyapunov_exponent_rosenstein, groundtruth)
    max_lyap_pred = safe_call(max_lyapunov_exponent_rosenstein, predictions)
    max_lyap_pred_with_context = safe_call(
        max_lyapunov_exponent_rosenstein, np.concatenate([context, predictions], axis=0)
    )
    max_lyap_gt_with_context = safe_call(
        max_lyapunov_exponent_rosenstein, np.concatenate([context, groundtruth], axis=0)
    )

    avg_hellinger_pred_horizon = safe_call(
        average_hellinger_distance, groundtruth, predictions
    )
    kl_pred_horizon = safe_call(estimate_kl_divergence, groundtruth, predictions)
    avg_hellinger_full_traj = safe_call(
        average_hellinger_distance, full_trajectory, predictions
    )
    kl_full_traj = safe_call(estimate_kl_divergence, full_trajectory, predictions)

    return dyst_name, {
        "prediction_horizon": {
            "avg_hellinger_distance": avg_hellinger_pred_horizon,
            "kl_divergence": kl_pred_horizon,
        },
        "full_trajectory": {
            "avg_hellinger_distance": avg_hellinger_full_traj,
            "kl_divergence": kl_full_traj,
        },
        "max_lyap_rosenstein": {
            "max_lyap_gt": max_lyap_gt,
            "max_lyap_gt_with_context": max_lyap_gt_with_context,
            "max_lyap_pred": max_lyap_pred,
            "max_lyap_pred_with_context": max_lyap_pred_with_context,
        },
        "prediction_time": elapsed_time,
    }


def get_distributional_metrics(
    forecast_dict: dict[str, dict[str, np.ndarray]],
    n_jobs: int | None = None,
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
                - "context": np.ndarray, context data of shape (d, T) or (T, d)
                - "predictions": np.ndarray, model predictions of shape (d, T) or (T, d)
                - "groundtruth": np.ndarray, ground truth data of shape (d, T) or (T, d)
                - "full_trajectory": np.ndarray, full trajectory data of shape (d, T) or (T, d)
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
    for dyst_name, data in forecast_dict.items():
        if not all(key in data for key in required_keys):
            raise ValueError(f"Missing required data for {dyst_name}: {required_keys}")

    results_all_pred_intervals = {}
    pred_intervals = [128, 192, 256, 320, 384, 448, 512]

    for pred_interval in pred_intervals:
        # Prepare arguments for parallel processing
        worker_args = [
            (dyst_name, data, pred_interval)
            for dyst_name, data in forecast_dict.items()
        ]
        # Use multiprocessing to compute dimensions in parallel
        with Pool(processes=n_jobs) as pool:
            results = list(
                tqdm(
                    # pool.imap(_compute_metrics_worker, worker_args),
                    pool.imap(_compute_more_metrics_worker, worker_args),
                    # pool.imap(_compute_dataset_metrics_worker, worker_args),
                    total=len(worker_args),
                    desc="Computing distributional metrics",
                )
            )
        results_all_pred_intervals[pred_interval] = results

    print(results_all_pred_intervals)
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
    pred = (
        model.predict(context_tensor, prediction_length, **kwargs)
        .squeeze()
        .cpu()
        .numpy()
    )
    elapsed_time = time.time() - start_time

    if is_chronos:
        if not kwargs.get("deterministic", False):
            pred = np.median(pred, axis=1)
    else:
        pred = pred.T

    return pred, elapsed_time


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    metrics_save_dir = cfg.eval.metrics_save_dir
    os.makedirs(metrics_save_dir, exist_ok=True)

    forecasts_dict_path = os.path.join(
        metrics_save_dir, f"{cfg.eval.metrics_fname}_forecasts.pkl"
    )

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
                "num_samples": 5,
                "deterministic": False,
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

        log(
            f"Using context length: {context_length} and prediction length: {prediction_length}"
        )

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
                coords, _ = zip(
                    *[(coord["target"], coord["start"]) for coord in file_dataset]
                )
                coordinates = np.stack(coords)
                if coordinates.ndim > 2:  # if not one_dim_target:
                    coordinates = coordinates.squeeze()

                context = coordinates[:, window_start_time:window_end_time]
                groundtruth = coordinates[
                    :, window_end_time : window_end_time + prediction_length
                ]

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
                forecast_dict = {
                    k: v for k, v in forecast_dict.items() if k != "full_trajectory"
                }
            with open(forecasts_dict_path, "wb") as f:
                pickle.dump(forecast_dict, f)

    else:
        log(f"Reloading forecasts from {forecasts_dict_path}")
        with open(forecasts_dict_path, "rb") as f:
            forecast_dict = pickle.load(f)

    distributional_metrics = get_distributional_metrics(
        forecast_dict, n_jobs=cfg.eval.num_processes
    )

    metrics_fname = (
        f"{cfg.eval.metrics_fname}_more"
        if cfg.eval.reload_saved_forecasts
        else cfg.eval.metrics_fname
    )
    metrics_path = os.path.join(metrics_save_dir, f"{metrics_fname}.json")
    log(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(distributional_metrics, f, indent=4)


if __name__ == "__main__":
    main()
