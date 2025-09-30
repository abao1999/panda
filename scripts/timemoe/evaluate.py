import logging
from functools import partial

import hydra
import numpy as np
import torch
import transformers
from gluonts.transform import LastValueImputation
from transformers import AutoModelForCausalLM

from panda.dataset import UnivariateTimeSeriesDataset
from panda.evaluation import evaluate_univariate_forecasting_model
from panda.utils.data_utils import get_dim_from_dataset, process_trajs
from panda.utils.eval_utils import get_eval_data_dict, save_evaluation_results
from panda.utils.train_utils import log_on_main

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


class TimeMoePipeline:
    def __init__(self, model_path: str, device: str, torch_dtype: torch.dtype):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        num_samples: int = 1,
        limit_prediction_length: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        seqs = context.to(self.device)
        mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
        normed_seqs = (seqs - mean) / std
        # shape: (batch_size, context_length + prediction_length)
        output = self.model.generate(normed_seqs, max_new_tokens=prediction_length)
        # shape: (batch_size, prediction_length)
        normed_predictions = output[:, -prediction_length:]
        return (normed_predictions * std + mean).unsqueeze(1)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    # init model for inference
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)
    pipeline = TimeMoePipeline("Maple728/TimeMoE-50M", device=cfg.eval.device, torch_dtype=torch_dtype)
    pipeline.model.eval()

    train_config = dict(cfg.train)

    # set floating point precision
    use_tf32 = train_config.get("tf32", False)
    log(f"use tf32: {use_tf32}")
    if use_tf32 and not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8):
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log(
            "TF32 format is only available on devices with compute capability >= 8. Setting tf32 to False.",
        )
        use_tf32 = False

    rseed = train_config.get("seed", cfg.train.seed)
    log(f"Using SEED: {rseed}")
    transformers.set_seed(seed=rseed)

    context_length = cfg.chronos.context_length
    prediction_length = cfg.eval.prediction_length
    log(f"context_length: {context_length}")
    log(f"model prediction_length: {prediction_length}")
    log(f"eval prediction_length: {cfg.eval.prediction_length}")

    # for convenience, get system dimensions
    system_dims = {system_name: get_dim_from_dataset(test_data_dict[system_name][0]) for system_name in test_data_dict}
    n_system_samples = {system_name: len(test_data_dict[system_name]) for system_name in test_data_dict}

    log(f"Running evaluation on {list(test_data_dict.keys())}")

    test_datasets = {
        system_name: UnivariateTimeSeriesDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])] * len(test_data_dict[system_name]),
            tokenizer=None,  # type: ignore # not relevant for evaluation
            context_length=cfg.chronos.context_length,
            prediction_length=cfg.eval.prediction_length,  # NOTE: should match the forecast prediction length
            min_past=cfg.min_past,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            model_type=cfg.chronos.model_type,
            imputation_method=LastValueImputation() if cfg.chronos.model_type == "causal" else None,
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

    parallel_sample_reduction_fn = {
        "mean": lambda x: np.mean(x, axis=0),
        "median": lambda x: np.median(x, axis=0),
    }[cfg.eval.parallel_sample_reduction]

    predictions, contexts, labels, metrics = evaluate_univariate_forecasting_model(
        pipeline,  # type: ignore
        test_datasets,
        batch_size=cfg.eval.batch_size,
        prediction_length=cfg.eval.prediction_length,
        metric_names=cfg.eval.metric_names,
        system_dims=system_dims,
        return_predictions=cfg.eval.save_forecasts,
        return_contexts=cfg.eval.save_contexts,
        return_labels=cfg.eval.save_labels,
        parallel_sample_reduction_fn=parallel_sample_reduction_fn,
        prediction_kwargs=dict(
            limit_prediction_length=cfg.eval.limit_prediction_length,
            verbose=cfg.eval.verbose,
        ),
        eval_subintervals=[(0, i + 64) for i in range(0, cfg.eval.prediction_length, 64)],
    )
    save_eval_results_fn(metrics)

    if cfg.eval.save_forecasts and predictions is not None and contexts is not None:
        process_trajs_fn(
            cfg.eval.forecast_save_dir,
            {system: np.concatenate([contexts[system], predictions[system]], axis=1) for system in predictions},
        )

    if cfg.eval.save_labels and labels is not None and contexts is not None:
        process_trajs_fn(
            cfg.eval.labels_save_dir,
            {system: np.concatenate([contexts[system], labels[system]], axis=1) for system in labels},
        )


if __name__ == "__main__":
    main()
