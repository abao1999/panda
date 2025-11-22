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
import transformers
from dysts.analysis import gp_dim  # type: ignore
from tqdm import tqdm

from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils.eval_utils import get_eval_data_dict
from panda.utils.train_utils import log_on_main

logger = logging.getLogger(__name__)
log = partial(log_on_main, logger=logger)


def _compute_gp_dims_worker(args) -> tuple[str, dict[str, float]]:
    """Worker function to compute GP dimensions for a single system"""
    dyst_name, data = args

    # Transpose arrays once
    completions_T = data["completions"].T
    groundtruth_T = data["processed_context"].T
    # timestep_mask = data["timestep_mask"]

    result = {
        "groundtruth": gp_dim(groundtruth_T),
        "completions": gp_dim(completions_T),
    }

    del completions_T, groundtruth_T
    gc.collect()

    return dyst_name, result


def get_gp_dims(
    completions_dict: dict[str, dict[str, np.ndarray]],
    n_jobs: int | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute GP dimensions for multiple systems using multiprocessing.

    Args:
        completions_dict: Dictionary containing completions and processed_context for each system
        n_jobs: Number of processes to use. If None, uses all available CPU cores

    Returns:
        Dictionary containing GP dimensions for groundtruth and completions for each system
    """
    # Validate all data upfront
    for dyst_name, data in completions_dict.items():
        if "completions" not in data or "processed_context" not in data:
            raise ValueError(f"Missing required data for {dyst_name}")

    # Prepare arguments for parallel processing
    worker_args = [(dyst_name, data) for dyst_name, data in completions_dict.items()]

    # Use multiprocessing to compute dimensions in parallel
    # maxtasksperchild prevents memory buildup in worker processes
    with Pool(processes=n_jobs, maxtasksperchild=10) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_compute_gp_dims_worker, worker_args, chunksize=1),
                total=len(worker_args),
                desc="Computing GP dimensions",
            )
        )

    # Explicitly delete worker_args to free memory
    del worker_args
    gc.collect()

    # Convert results list back to dictionary
    return dict(results)


def get_model_completion(
    pipeline,
    context: np.ndarray,
    return_normalized_completions: bool = False,
    verbose: bool = True,
    **kwargs,
):
    # Prepare input tensor
    context_tensor = torch.from_numpy(context.T).float().to(pipeline.device)[None, ...]
    # Generate completions
    completions_output = pipeline.model.generate_completions(
        context_tensor,
        past_observed_mask=None,
        **kwargs,
    )

    if verbose:
        print(f"context_tensor shape: {context_tensor.shape}")
        print(f"completions output shape: {completions_output.completions.shape}")

    # Extract shapes and data
    patch_size = completions_output.completions.shape[-1]

    # Check for required outputs
    if any(x is None for x in [completions_output.mask, completions_output.patched_past_values]):
        raise ValueError("Required completion outputs are None")

    # Process tensors to numpy arrays
    def process_tensor(tensor, reshape=True):
        if reshape:
            return (
                tensor.reshape(context_tensor.shape[0], context_tensor.shape[-1], -1)
                .detach()
                .cpu()
                .numpy()
                .transpose(0, 2, 1)
            )
        return tensor.detach().cpu().numpy()

    completions = process_tensor(completions_output.completions)
    processed_context = process_tensor(completions_output.patched_past_values)
    patch_mask = process_tensor(completions_output.mask, reshape=False)
    timestep_mask = np.repeat(patch_mask, repeats=patch_size, axis=2)

    # Denormalize if needed
    if not return_normalized_completions:
        if completions_output.loc is None or completions_output.scale is None:
            raise ValueError("Loc or scale is None")
        loc = completions_output.loc.detach().cpu().numpy()
        scale = completions_output.scale.detach().cpu().numpy()
        completions = completions * scale + loc
        processed_context = processed_context * scale + loc

    # Reshape for plotting
    processed_context = processed_context.squeeze(0).transpose(1, 0)
    completions = completions.squeeze(0).transpose(1, 0)
    timestep_mask = timestep_mask.squeeze(0)

    if verbose:
        print(f"processed context shape: {processed_context.shape}")
        print(f"completions shape: {completions.shape}")
        print(f"timestep mask shape: {timestep_mask.shape}")

    return completions, processed_context, timestep_mask


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    test_data_dict = get_eval_data_dict(
        cfg.eval.data_paths_lst,
        num_subdirs=cfg.eval.num_subdirs,
        num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
    )
    log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

    # Load MLM checkpoint
    checkpoint_path = cfg.eval.checkpoint_path
    log(f"Using checkpoint: {checkpoint_path}")
    rseed = cfg.eval.seed
    log(f"Using SEED: {rseed}")
    transformers.set_seed(seed=rseed)

    model_pipeline = PatchTSTPipeline.from_pretrained(
        mode="pretrain",
        pretrain_path=checkpoint_path,
        device_map=cfg.eval.device,
        torch_dtype=getattr(torch, cfg.eval.torch_dtype, torch.float32),
    )

    start_time = cfg.eval.completions.start_time
    end_time = cfg.eval.completions.end_time

    log(f"Using context from {start_time} to {end_time}")

    completions_dict = {}
    for subdir_name, datasets in tqdm(
        list(test_data_dict.items())[: cfg.eval.num_subdirs],
        desc="Generating completions for subdirectories",
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

            completions, processed_context, timestep_mask = get_model_completion(
                model_pipeline,
                coordinates[:, start_time:end_time],  # context
                return_normalized_completions=False,
                verbose=False,
            )
            completions_dict[system_name] = {
                "completions": completions,
                "processed_context": processed_context,
                "timestep_mask": timestep_mask,
            }

    gp_dims = get_gp_dims(completions_dict, n_jobs=cfg.eval.num_processes)

    metrics_save_dir = cfg.eval.metrics_save_dir
    metrics_fname = f"{cfg.eval.metrics_fname}"
    metrics_path = os.path.join(metrics_save_dir, f"{metrics_fname}_start{start_time}_end{end_time}_rseed{rseed}.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    log(f"Saving GP dimensions to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(gp_dims, f, indent=4)

    if cfg.eval.save_completions:
        completions_save_path = os.path.join(
            metrics_save_dir, f"{metrics_fname}_completions_start{start_time}_end{end_time}_rseed{rseed}.pkl"
        )
        os.makedirs(os.path.dirname(completions_save_path), exist_ok=True)

        log(f"Saving completions to {completions_save_path}")
        with open(completions_save_path, "wb") as f:
            pickle.dump(completions_dict, f)
        log(f"Saved completions to {completions_save_path}")


if __name__ == "__main__":
    main()
