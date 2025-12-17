import logging
from functools import partial

import hydra
import numpy as np
import torch
import transformers

from panda.baselines.fm_baselines import DynaMixPipeline
from panda.dataset import MultivariateTimeSeriesDataset
from panda.evaluation import evaluate_multivariate_forecasting_model
from panda.utils.data_utils import get_dim_from_dataset, process_trajs
from panda.utils.eval_utils import get_eval_data_dict, save_evaluation_results
from panda.utils.train_utils import log_on_main

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    save_forecasts = cfg.eval.save_forecasts
    save_contexts = cfg.eval.save_contexts
    save_labels = cfg.eval.save_labels

    forecast_save_dir = cfg.eval.forecast_save_dir
    labels_save_dir = cfg.eval.labels_save_dir

    # NOTE: we fail loudly here so the user is aware that they need to specify the save directories if they want to save the forecasts or labels
    if save_forecasts and forecast_save_dir is None:
        raise ValueError("forecast_save_dir must be specified if save_forecasts is True")
    if save_labels and labels_save_dir is None:
        raise ValueError("labels_save_dir must be specified if save_labels is True")

    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    torch_dtype = getattr(torch, cfg.dynamix.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
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
    transformers.set_seed(seed=rseed)

    context_length = cfg.chronos.context_length
    prediction_length = cfg.eval.prediction_length
    log(f"context_length: {context_length}")
    log(f"prediction_length: {prediction_length}")

    system_dims = {system_name: get_dim_from_dataset(test_data_dict[system_name][0]) for system_name in test_data_dict}
    n_system_samples = {system_name: len(test_data_dict[system_name]) for system_name in test_data_dict}

    log(f"Running evaluation on {list(test_data_dict.keys())}")

    test_datasets = {
        system_name: MultivariateTimeSeriesDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])] * len(test_data_dict[system_name]),
            context_length=context_length,
            prediction_length=cfg.eval.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            model_type="predict",
            mode="test",
        )
        for system_name in test_data_dict
    }

    save_eval_results_fn = partial(
        save_evaluation_results,
        metrics_metadata={
            "system_dims": system_dims,
            "n_system_samples": n_system_samples,
        },
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

    parallel_sample_reduction_fn = {
        "mean": lambda x: np.mean(x, axis=0),
        "median": lambda x: np.median(x, axis=0),
    }.get(cfg.eval.parallel_sample_reduction, lambda x: x)

    predictions, contexts, labels, metrics = evaluate_multivariate_forecasting_model(
        pipeline,  # type: ignore[arg-type]
        test_datasets,
        batch_size=cfg.eval.batch_size,
        prediction_length=cfg.eval.prediction_length,
        system_dims=system_dims,
        metric_names=cfg.eval.metric_names,
        return_predictions=save_forecasts,
        return_contexts=save_contexts,
        return_labels=save_labels,
        parallel_sample_reduction_fn=parallel_sample_reduction_fn,
        redo_normalization=True,
        prediction_kwargs={},
        eval_subintervals=[(0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)],
        num_workers=cfg.eval.dataloader_num_workers,
    )
    save_eval_results_fn(metrics)

    if save_forecasts:
        assert predictions is not None
        if save_contexts:
            assert contexts is not None
            process_trajs_fn(
                forecast_save_dir,
                {system: np.concatenate([contexts[system], predictions[system]], axis=2) for system in predictions},
            )
        else:
            process_trajs_fn(
                forecast_save_dir,
                {system: predictions[system] for system in predictions},
            )

    if save_labels:
        assert labels is not None
        if save_contexts:
            assert contexts is not None
            process_trajs_fn(
                labels_save_dir,
                {system: np.concatenate([contexts[system], labels[system]], axis=2) for system in labels},
            )
        else:
            process_trajs_fn(
                labels_save_dir,
                {system: labels[system] for system in labels},
            )


if __name__ == "__main__":
    main()
