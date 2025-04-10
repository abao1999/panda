"""
Analyze a dataset of pre-computed trajectories, loading from either Arrow files or npy files
"""

import logging
import os
from collections import defaultdict

import hydra
import numpy as np
from dysts.metrics import compute_metrics  # type: ignore

from dystformer.utils import make_ensemble_from_arrow_dir, save_evaluation_results


def compute_metrics_from_combined_ensemble(
    combined_ensemble: dict[str, dict[str, np.ndarray]],
    metric_names: list[str] | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
    context_length: int = 0,
    prediction_length: int = 0,
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Compute metrics from a combined ensemble of forecasts and labels
    Args:
        combined_ensemble: Dict[str, Dict[str, np.ndarray]]
            Dictionary with key being system name, whose entries are dictionaries with keys "forecasts" and "labels"
            and values being numpy arrays of shape (num_samples, dim, context_length + forecast_length)
    """

    system_metrics = defaultdict(dict)
    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))

    for system_name, system_data in combined_ensemble.items():
        forecasts_with_context = system_data["forecasts"].transpose(0, 2, 1)
        labels_with_context = system_data["labels"].transpose(0, 2, 1)
        forecasts = forecasts_with_context[:, context_length:, :]
        labels = labels_with_context[:, context_length:, :]

        # evaluate metrics for multiple forecast lengths on user-specified subintervals
        # as well as the full prediction length interval
        if metric_names is not None:
            assert all(start < prediction_length for start, _ in eval_subintervals), (
                "All start indices must be less than the prediction length"
            )
            for start, end in eval_subintervals:
                system_metrics[end - start][system_name] = compute_metrics(
                    forecasts[:, start:end, :],
                    labels[:, start:end, :],
                    include=metric_names,
                    batch_axis=0,
                )

    return system_metrics


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    save_dir = cfg.recompute_metrics.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # make ensemble from saved trajectories in Arrow files
    forecast_ensemble = make_ensemble_from_arrow_dir(
        cfg.recompute_metrics.eval_data_dir,
        cfg.recompute_metrics.forecast_split,
        one_dim_target=False,
        num_samples=cfg.recompute_metrics.num_samples,
    )
    logger.info(
        f"Loaded {len(forecast_ensemble)} systems from {cfg.recompute_metrics.eval_data_dir} split {cfg.recompute_metrics.forecast_split}"
    )

    # Now load the labels ensemble
    labels_ensemble = make_ensemble_from_arrow_dir(
        cfg.recompute_metrics.eval_data_dir,
        cfg.recompute_metrics.labels_split,
        one_dim_target=False,
        num_samples=cfg.recompute_metrics.num_samples,
    )
    logger.info(
        f"Loaded {len(labels_ensemble)} systems from {cfg.recompute_metrics.eval_data_dir} split {cfg.recompute_metrics.labels_split}"
    )

    # now combine the forecast_ensemble and labels_ensemble into a single ensemble
    combined_ensemble = {}
    for system_name in forecast_ensemble.keys():
        if system_name not in labels_ensemble:
            raise ValueError(f"System {system_name} not found in labels ensemble!")
        sys_forecasts = forecast_ensemble[system_name]
        sys_labels = labels_ensemble[system_name]
        if sys_forecasts.shape != sys_labels.shape:
            raise ValueError(
                f"Forecast and labels for system {system_name} have different shapes!"
            )
        combined_ensemble[system_name] = {
            "forecasts": sys_forecasts,
            "labels": sys_labels,
        }

    logger.info(f"Combined ensemble has {len(combined_ensemble)} systems")

    breakpoint()

    eval_subintervals = []
    system_metrics = compute_metrics_from_combined_ensemble(
        combined_ensemble,
        metric_names=cfg.recompute_metrics.metric_names,
        eval_subintervals=cfg.recompute_metrics.eval_subintervals,
        context_length=cfg.recompute_metrics.context_length,
        prediction_length=cfg.recompute_metrics.prediction_length,
    )

    save_evaluation_results(
        system_metrics,
        coords=None,
        metrics_save_dir=save_dir,
        metrics_fname=cfg.recompute_metrics.metrics_fname,
        overwrite=True,
        coords_save_dir=None,  # don't save coords
        split_coords=False,
        verbose=True,
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
