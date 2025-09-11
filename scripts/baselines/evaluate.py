import logging
from functools import partial

import hydra
import numpy as np
import transformers

from panda.baselines.baselines import (
    FourierARIMABaseline,
    FourierBaseline,
    MeanBaseline,
)
from panda.baselines.evaluation import evaluate_forecasting_model
from panda.dataset import MultivariateTimeSeriesDataset
from panda.utils import (
    get_dim_from_dataset,
    get_eval_data_dict,
    log_on_main,
    process_trajs,
    save_evaluation_results,
)

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    transformers.set_seed(seed=cfg.eval.seed)
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
        system_name: MultivariateTimeSeriesDataset(
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

    save_eval_results_fn = partial(
        save_evaluation_results,
        metrics_metadata={
            "system_dims": system_dims,
            "n_system_samples": n_system_samples,
        },  # pass metadata to be saved as columns in metrics csv
        metrics_save_dir=cfg.eval.metrics_save_dir,
        metrics_fname=cfg.eval.metrics_fname,
        overwrite=cfg.eval.overwrite,
    )
    process_trajs_fn = partial(
        process_trajs,
        split_coords=cfg.eval.split_coords,
        overwrite=cfg.eval.overwrite,
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
    save_eval_results_fn(metrics)

    if cfg.eval.save_forecasts and predictions is not None and contexts is not None:
        process_trajs_fn(
            cfg.eval.forecast_save_dir,
            {
                system: np.concatenate([contexts[system], predictions[system]], axis=2)
                for system in predictions
            },
        )

    if cfg.eval.save_labels and labels is not None and contexts is not None:
        process_trajs_fn(
            cfg.eval.labels_save_dir,
            {
                system: np.concatenate([contexts[system], labels[system]], axis=2)
                for system in labels
            },
        )


if __name__ == "__main__":
    main()
