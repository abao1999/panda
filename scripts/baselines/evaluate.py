import logging
import os
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import transformers
from gluonts.dataset.common import FileDataset

from dystformer.baselines.baselines import (
    FourierARIMABaseline,
    FourierBaseline,
    MeanBaseline,
)
from dystformer.baselines.evaluation import evaluate_forecasting_model
from dystformer.patchtst.dataset import TimeSeriesDataset
from dystformer.utils import get_dim_from_dataset, log_on_main, save_evaluation_results

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    transformers.set_seed(seed=cfg.eval.seed)

    # get test data paths
    test_data_dir = os.path.expandvars(cfg.eval.data_path)
    test_data_dict = {}
    system_dirs = [d for d in Path(test_data_dir).iterdir() if d.is_dir()]

    for system_dir in system_dirs[: cfg.eval.num_systems]:
        system_name = system_dir.name
        system_files = list(system_dir.glob("*"))
        test_data_dict[system_name] = [
            FileDataset(path=Path(file_path), freq="h", one_dim_target=False)
            for file_path in system_files
            if file_path.is_file()
        ]

    # for convenience, get system dimensions, for saving as a column in the metrics csv
    system_dims = {
        system_name: get_dim_from_dataset(test_data_dict[system_name][0])
        for system_name in test_data_dict
    }
    n_system_samples = {
        system_name: len(test_data_dict[system_name]) for system_name in test_data_dict
    }

    log(f"Running evaluation on {list(test_data_dict.keys())}")

    test_datasets = {
        system_name: TimeSeriesDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])]
            * len(test_data_dict[system_name]),
            context_length=cfg.eval.context_length,
            prediction_length=cfg.eval.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            model_type=cfg.eval.mode,
            mode="test",
        )
        for system_name in test_data_dict
    }

    save_eval_results = partial(
        save_evaluation_results,
        metrics_metadata={
            "system_dims": system_dims,
            "n_system_samples": n_system_samples,
        },  # pass metadata to be saved as columns in metrics csv
        metrics_save_dir=cfg.eval.metrics_save_dir,
        metrics_fname=cfg.eval.metrics_fname,
        overwrite=cfg.eval.overwrite,
        split_coords=cfg.eval.split_coords,
        verbose=cfg.eval.verbose,
    )
    log(f"Saving evaluation results to {cfg.eval.metrics_save_dir}")

    baseline_model = {
        "fourier_arima": FourierARIMABaseline(
            prediction_length=cfg.eval.prediction_length,
            order=cfg.eval.baselines.order,
            num_fourier_terms=cfg.eval.baselines.num_fourier_terms,
        ),
        "mean": MeanBaseline(prediction_length=cfg.eval.prediction_length),
        "fourier": FourierBaseline(prediction_length=cfg.eval.prediction_length),
    }[cfg.eval.baselines.baseline_model]

    predictions, contexts, labels, metrics = evaluate_forecasting_model(
        baseline_model,
        test_datasets,
        batch_size=cfg.eval.batch_size,
        prediction_length=cfg.eval.prediction_length,
        metric_names=cfg.eval.metric_names,
        return_predictions=True,
        return_contexts=True,
        return_labels=True,
        redo_normalization=True,
        eval_subintervals=[
            (0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)
        ],
    )

    if predictions is not None and contexts is not None:
        full_trajs = {}
        for system in predictions:
            if system not in contexts:
                raise ValueError(f"System {system} not in contexts")
            # shape: (num_eval_windows*num_datasets, num_channels, context_length + prediction_length)
            full_trajs[system] = np.concatenate(
                [contexts[system], predictions[system]], axis=2
            )
        save_eval_results(
            metrics,
            coords=full_trajs,
            coords_save_dir=cfg.eval.forecast_save_dir,
        )

    if labels is not None and contexts is not None:
        full_trajs = {}
        for system in labels:
            if system not in contexts:
                raise ValueError(f"System {system} not in contexts")
            # shape: (num_eval_windows*num_datasets, num_channels, context_length + prediction_length)
            full_trajs[system] = np.concatenate(
                [contexts[system], labels[system]], axis=2
            )
        save_eval_results(
            None,  # do not save metrics again
            coords=full_trajs,
            coords_save_dir=cfg.eval.labels_save_dir,
        )


if __name__ == "__main__":
    main()
