"""
This script measures the inference time for a given model and dataset.
"""

import json
import logging
import os
import time
from functools import partial

import hydra
import numpy as np
import torch
import transformers
from tqdm import tqdm

from panda.baselines.fm_baselines import DynaMixPipeline, TimeMoePipeline, TimesFMPipeline
from panda.chronos.pipeline import ChronosPipeline
from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils.eval_utils import get_eval_data_dict
from panda.utils.train_utils import log_on_main

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)

UNIVARIATE_MODELS = ["chronos", "timesfm", "timemoe"]
MULTIVARIATE_MODELS = ["panda", "dynamix"]


def get_model_prediction(
    model,
    context: np.ndarray,
    prediction_length: int,
    is_multivariate: bool = False,
    prediction_kwargs: dict = {},
) -> float:
    """
    Generate model predictions for a given context and prediction length.

    Args:
        model: The model to use for prediction.
        context (np.ndarray): The input context array. Shape should be (dim, timesteps).
        prediction_length (int): The number of timesteps to predict.
        is_multivariate (bool, optional): If True, use multivariate model context processing convention. Default is False.
        **kwargs: Additional keyword arguments to pass to the model's predict method.

    Returns:
        tuple[np.ndarray, float]:
            - elapsed_time (float): The time taken for prediction in seconds.
    """
    # if univariate model, the context must be (dim, timesteps) where the first dimension is treated as batch dimension
    context_tensor = torch.from_numpy(context.T if is_multivariate else context).float()

    start_time = time.time()
    _ = model.predict(context_tensor, prediction_length, **prediction_kwargs).squeeze().cpu().numpy()
    elapsed_time = time.time() - start_time

    return elapsed_time


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

    model_prediction_kwargs = {
        "panda": {
            "limit_prediction_length": False,
            "sliding_context": True,
            "verbose": False,
        },
        "chronos": {
            "limit_prediction_length": False,
            "deterministic": True if cfg.eval.chronos.deterministic else False,
            "num_samples": 1 if cfg.eval.chronos.deterministic else cfg.eval.num_samples,
            "verbose": False,
        },
        "dynamix": {},
        "timesfm": {"limit_prediction_length": False, "verbose": False},
        "timemoe": {"limit_prediction_length": False, "verbose": False},
    }[cfg.eval.model_type]

    is_multivariate = False
    if cfg.eval.model_type == "panda":
        log(f"Using PatchTST checkpoint: {cfg.eval.checkpoint_path}")
        model_pipeline = PatchTSTPipeline.from_pretrained(
            mode="predict",
            pretrain_path=cfg.eval.checkpoint_path,
            device_map=cfg.eval.device,
            torch_dtype=getattr(torch, cfg.eval.torch_dtype, torch.float32),
        )
    elif cfg.eval.model_type == "chronos":
        log(f"Using Chronos checkpoint: {cfg.eval.checkpoint_path}")
        model_pipeline = ChronosPipeline.from_pretrained(
            cfg.eval.checkpoint_path,
            device_map=cfg.eval.device,
            torch_dtype=getattr(torch, cfg.eval.torch_dtype, torch.float32),
        )
    elif cfg.eval.model_type == "timesfm":
        log(f"Using TimesFM checkpoint: {cfg.eval.checkpoint_path}")
        model_pipeline = TimesFMPipeline(
            model_id=cfg.eval.checkpoint_path,
            device=cfg.eval.device,
            prediction_length=cfg.eval.prediction_length,
        )
    elif cfg.eval.model_type == "timemoe":
        log(f"Using TimeMoE checkpoint: {cfg.eval.checkpoint_path}")
        model_pipeline = TimeMoePipeline(
            model_path=cfg.eval.checkpoint_path,
            device=cfg.eval.device,
            torch_dtype=getattr(torch, cfg.eval.torch_dtype, torch.float32),
        )
    elif cfg.eval.model_type == "dynamix":
        log(f"Using Dynamix model (cfg.dynamix.model_name): {cfg.dynamix.model_name}")
        model_pipeline = DynaMixPipeline(
            model_name=cfg.dynamix.model_name,
            device=cfg.dynamix.device,
            torch_dtype=getattr(torch, cfg.eval.torch_dtype, torch.float32),
            preprocessing_method=cfg.dynamix.preprocessing_method,
            standardize=cfg.dynamix.standardize,
            fit_nonstationary=cfg.dynamix.fit_nonstationary,
        )
        train_config = dict(cfg.train)
        rseed = train_config.get("seed", cfg.train.seed)
        log(f"Using SEED: {rseed}")
        transformers.set_seed(seed=rseed)
    else:
        raise ValueError(f"Invalid model type: {cfg.eval.model_type}")

    is_multivariate = cfg.eval.model_type in MULTIVARIATE_MODELS

    prediction_length = cfg.eval.prediction_length
    context_length = cfg.eval.context_length
    window_start_time = cfg.eval.window_start
    window_end_time = window_start_time + context_length

    log(f"Timing inference for {cfg.eval.model_type} model from checkpoint: {cfg.eval.checkpoint_path}")
    log(f"Using context length: {context_length} and prediction length: {prediction_length}")
    log(f"Using window start time: {window_start_time} and window end time: {window_end_time}")
    log(f"Using is_multivariate: {is_multivariate}")
    log(f"Using model kwargs: {model_prediction_kwargs}")
    log(f"Using num_samples (Chronos): {cfg.eval.num_samples}")
    log(f"Using use_deterministic_chronos: {cfg.eval.chronos.deterministic}")
    log(f"Using num_samples_per_subdir: {cfg.eval.num_samples_per_subdir}")
    log(f"Using num_subdirs: {cfg.eval.num_subdirs}")

    elapsed_time_dict = {}
    for subdir_name, datasets in tqdm(
        list(test_data_dict.items())[: cfg.eval.num_subdirs],
        desc="Generating forecasts for subdirectories",
    ):
        log(f"Processing {len(datasets)} datasets in {subdir_name}")
        elapsed_times = []
        for file_dataset in datasets[: cfg.eval.num_samples_per_subdir]:
            coords, _ = zip(*[(coord["target"], coord["start"]) for coord in file_dataset])
            coordinates = np.stack(coords)
            if coordinates.ndim > 2:  # if not one_dim_target:
                coordinates = coordinates.squeeze()

            context = coordinates[:, window_start_time:window_end_time]

            elapsed_time = get_model_prediction(
                model_pipeline,
                context=context,
                prediction_length=prediction_length,
                is_multivariate=is_multivariate,
                prediction_kwargs=model_prediction_kwargs,
            )
            elapsed_times.append(elapsed_time)

        elapsed_time_dict[subdir_name] = elapsed_times

    # NOTE: we remove the first time estimate because it is skewed by loadiing the checkpoint and I/O overhead
    elapsed_time_dict[list(elapsed_time_dict.keys())[0]].pop(0)
    log(f"Elapsed time dictionary: {elapsed_time_dict}")
    log(f"Saving elapsed time dictionary to {metrics_save_dir}")
    all_elapsed_times = [elapsed_time for subdir_name in elapsed_time_dict.values() for elapsed_time in subdir_name]
    average_elapsed_time = np.mean(all_elapsed_times)
    std_elapsed_time = np.std(all_elapsed_times)
    log(f"Average elapsed time (over {len(all_elapsed_times)} values): {average_elapsed_time}")
    log(f"Standard deviation of elapsed time (over {len(all_elapsed_times)} values): {std_elapsed_time}")
    inference_times_fname = cfg.eval.metrics_fname + f"_predlength{prediction_length}.json"
    with open(os.path.join(metrics_save_dir, inference_times_fname), "w") as f:
        json.dump(elapsed_time_dict, f)


if __name__ == "__main__":
    main()
