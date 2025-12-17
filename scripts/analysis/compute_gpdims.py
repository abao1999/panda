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
from scipy.interpolate import interp1d, make_interp_spline
from tqdm import tqdm

from panda.patchtst.pipeline import PatchTSTPipeline
from panda.utils.eval_utils import get_eval_data_dict
from panda.utils.plot_utils import plot_model_completion
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


################################################################################
# Naive interpolation baselines (e.g. polynomial interpolation for the masked-out timesteps)
################################################################################


def polynomial_interpolation(
    processed_context: np.ndarray,
    timestep_mask: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """
    Perform polynomial interpolation on masked timesteps for each dimension independently.

    For each dimension, this function identifies which timesteps are masked (True in timestep_mask)
    and interpolates their values by fitting a polynomial to the unmasked timesteps, then
    evaluating it at the masked positions. This allows different dimensions to have different
    masked timesteps.

    Args:
        processed_context: Array of shape (T, d) containing the time series data,
                          where T is the number of timesteps and d is the number of dimensions.
        timestep_mask: Boolean array of shape (T, d) indicating which values to interpolate.
                      True values will be interpolated from False (known) values.
        degree: Degree of the polynomial to fit. Default is 3 (cubic polynomial).
               Will be automatically reduced if there are insufficient known points.

    Returns:
        Array of shape (T, d) with interpolated values at masked timesteps.
        Known (unmasked) values are preserved from the input.

    Notes:
        - Uses numpy.polyfit to fit polynomials and numpy.polyval to evaluate them.
        - If the number of known points is less than degree + 1, the degree is
          automatically reduced to n_known - 1.
        - If all timesteps in a dimension are masked, the function will fail.
        - If only one timestep is unmasked in a dimension, it will use a constant
          (degree 0) polynomial.

    Raises:
        ValueError: If shapes don't match or if all timesteps in a dimension are masked.
    """
    if processed_context.shape != timestep_mask.shape:
        raise ValueError(
            f"Shape mismatch: processed_context has shape {processed_context.shape}, "
            f"but timestep_mask has shape {timestep_mask.shape}. Both should be (T, d)."
        )

    T, d = processed_context.shape
    result = np.zeros_like(processed_context)

    # Interpolate each dimension independently
    for dim in range(d):
        # Get indices of known (unmasked) and unknown (masked) timesteps for this dimension
        dim_mask = timestep_mask[:, dim]
        known_indices = np.where(dim_mask)[0]
        unknown_indices = np.where(~dim_mask)[0]

        # Copy known values directly
        result[known_indices, dim] = processed_context[known_indices, dim]

        # Interpolate unknown values if there are any
        if len(unknown_indices) > 0 and len(known_indices) > 0:
            # Adjust polynomial degree if we don't have enough points
            effective_degree = min(degree, len(known_indices) - 1)

            # Fit polynomial to known points
            # polyfit returns coefficients in descending order (highest degree first)
            coefficients = np.polyfit(known_indices, processed_context[known_indices, dim], deg=effective_degree)

            # Evaluate polynomial at unknown indices
            result[unknown_indices, dim] = np.polyval(coefficients, unknown_indices)
        elif len(unknown_indices) > 0 and len(known_indices) == 0:
            # All values are masked for this dimension - cannot interpolate
            raise ValueError(f"All timesteps are masked for dimension {dim}. Cannot interpolate.")

    return result


def linear_interpolation(
    processed_context: np.ndarray,
    timestep_mask: np.ndarray,
) -> np.ndarray:
    """
    Perform linear interpolation on masked timesteps for each dimension independently.

    For each dimension, this function identifies which timesteps are masked (True in timestep_mask)
    and interpolates their values using linear interpolation from the unmasked timesteps.
    This allows different dimensions to have different masked timesteps.

    Args:
        processed_context: Array of shape (T, d) containing the time series data,
                          where T is the number of timesteps and d is the number of dimensions.
        timestep_mask: Boolean array of shape (T, d) indicating which values to are known.
                      True values are known, False values are masked.

    Returns:
        Array of shape (T, d) with interpolated values at masked timesteps.
        Known (unmasked) values are preserved from the input.

    Notes:
        - Uses scipy.interpolate.interp1d with linear interpolation and extrapolation
          for timesteps outside the range of known values.
        - If all timesteps in a dimension are masked, the function will fail.
        - If only one timestep is unmasked in a dimension, extrapolation will use
          a constant value.
    """
    if processed_context.shape != timestep_mask.shape:
        raise ValueError(
            f"Shape mismatch: processed_context has shape {processed_context.shape}, "
            f"but timestep_mask has shape {timestep_mask.shape}. Both should be (T, d)."
        )

    T, d = processed_context.shape
    result = np.zeros_like(processed_context)

    # Interpolate each dimension independently
    for dim in range(d):
        # Get indices of known (unmasked) and unknown (masked) timesteps for this dimension
        dim_mask = timestep_mask[:, dim]
        known_indices = np.where(dim_mask)[0]
        unknown_indices = np.where(~dim_mask)[0]

        # Copy known values directly
        result[known_indices, dim] = processed_context[known_indices, dim]

        # Interpolate unknown values if there are any
        if len(unknown_indices) > 0 and len(known_indices) > 0:
            f = interp1d(
                known_indices,
                processed_context[known_indices, dim],
                kind="linear",
                fill_value="extrapolate",  # type: ignore
            )
            result[unknown_indices, dim] = f(unknown_indices)
        elif len(unknown_indices) > 0 and len(known_indices) == 0:
            # All values are masked for this dimension - cannot interpolate
            raise ValueError(f"All timesteps are masked for dimension {dim}. Cannot interpolate.")

    return result


def piecewise_spline_interpolation(
    processed_context: np.ndarray,
    timestep_mask: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Perform piecewise spline interpolation on masked timesteps for each dimension independently.

    For each dimension, this function identifies which timesteps are masked (True in timestep_mask)
    and interpolates their values using spline interpolation from the unmasked timesteps.
    This allows different dimensions to have different masked timesteps.

    Args:
        processed_context: Array of shape (T, d) containing the time series data,
                          where T is the number of timesteps and d is the number of dimensions.
        timestep_mask: Boolean array of shape (T, d) indicating which values to are known.
                      True values are known, False values are masked.
        k: Degree of the spline interpolation. Default is 3 (cubic spline).
           Must have at least k+1 known points to fit a spline of degree k.

    Returns:
        Array of shape (T, d) with interpolated values at masked timesteps.
        Known (unmasked) values are preserved from the input.

    Notes:
        - Uses scipy.interpolate.make_interp_spline for spline fitting.
        - Extrapolation is enabled for timesteps outside the range of known values.
        - If there are fewer than k+1 known points in a dimension, the degree
          is automatically reduced to max(len(known_indices) - 1, 1).
        - If all timesteps in a dimension are masked, the function will fail.
        - If only one timestep is unmasked in a dimension, linear extrapolation
          (constant value) is used.
    """
    if processed_context.shape != timestep_mask.shape:
        raise ValueError(
            f"Shape mismatch: processed_context has shape {processed_context.shape}, "
            f"but timestep_mask has shape {timestep_mask.shape}. Both should be (T, d)."
        )

    T, d = processed_context.shape
    result = np.zeros_like(processed_context)

    # Interpolate each dimension independently
    for dim in range(d):
        # Get indices of known (unmasked) and unknown (masked) timesteps for this dimension
        dim_mask = timestep_mask[:, dim]
        known_indices = np.where(dim_mask)[0]
        unknown_indices = np.where(~dim_mask)[0]

        # Copy known values directly
        result[known_indices, dim] = processed_context[known_indices, dim]

        # Interpolate unknown values if there are any
        if len(unknown_indices) > 0 and len(known_indices) > 0:
            # Determine appropriate spline degree based on available points
            # Need at least k+1 points for degree k spline
            effective_k = min(k, len(known_indices) - 1)
            effective_k = max(effective_k, 1)  # At least linear

            # Create spline interpolation
            spline = make_interp_spline(
                known_indices, processed_context[known_indices, dim], k=effective_k, bc_type="natural"
            )
            result[unknown_indices, dim] = spline(unknown_indices)
        elif len(unknown_indices) > 0 and len(known_indices) == 0:
            # All values are masked for this dimension - cannot interpolate
            raise ValueError(f"All timesteps are masked for dimension {dim}. Cannot interpolate.")

    return result


def get_naive_interpolation(
    completions_dict: dict[str, dict[str, np.ndarray]],
    interpolation_method: str = "polynomial",
    polynomial_degree: int = 3,
    piecewise_spline_degree: int = 3,
) -> dict[str, np.ndarray]:
    """
    Compute naive interpolations for multiple systems, using the timestep_mask and context from the completions_dict

    Args:
        completions_dict: Dictionary containing completions and processed_context for each system
        interpolation_method: Method to use for interpolation. Options: "polynomial", "linear"
        polynomial_degree: Degree of polynomial to use if interpolation_method is "polynomial". Default is 3.
        use_multiprocessing: If True, use multiprocessing to parallelize computation across systems. Default is False.
        n_jobs: Number of processes to use when use_multiprocessing=True. If None, uses all available CPU cores.

    Returns:
        Dictionary containing naive interpolations for each system (dyst_name -> interpolated array)
    """
    interpolation_fn = {
        "polynomial": partial(polynomial_interpolation, degree=polynomial_degree),
        "linear": linear_interpolation,
        "piecewise_spline": partial(piecewise_spline_interpolation, k=piecewise_spline_degree),
    }[interpolation_method]

    # Validate all data upfront
    for dyst_name, data in completions_dict.items():
        if "processed_context" not in data or "timestep_mask" not in data:
            raise ValueError(f"Missing timestep_mask or processed_context for {dyst_name}")

    log(f"Computing {interpolation_method} interpolations sequentially")
    # Sequential processing
    naive_interpolations = {}
    iterator = tqdm(completions_dict.items(), desc=f"Computing {interpolation_method} interpolations")
    for dyst_name, data in iterator:
        # Ensure (T, d) shape for all arrays
        processed_context = data["processed_context"].T
        timestep_mask = data["timestep_mask"].T

        naive_interpolations[dyst_name] = interpolation_fn(
            processed_context,
            timestep_mask,
        ).T

    return naive_interpolations


################################################################################


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    metrics_save_dir = cfg.eval.metrics_save_dir
    metrics_fname = cfg.eval.metrics_fname
    os.makedirs(metrics_save_dir, exist_ok=True)

    start_time = cfg.eval.completions.start_time
    end_time = cfg.eval.completions.end_time
    rseed = cfg.eval.seed

    completions_dict_path = os.path.join(
        metrics_save_dir, f"{metrics_fname}_completions_start{start_time}_end{end_time}_rseed{rseed}.pkl"
    )
    if cfg.eval.reload_saved_completions:
        log(f"Reloading saved completions from {completions_dict_path}")
        with open(completions_dict_path, "rb") as f:
            completions_dict = pickle.load(f)

    else:
        test_data_dict = get_eval_data_dict(
            cfg.eval.data_paths_lst,
            num_subdirs=cfg.eval.num_subdirs,
            num_samples_per_subdir=cfg.eval.num_samples_per_subdir,
        )
        log(f"Number of combined test data subdirectories: {len(test_data_dict)}")

        # Load MLM checkpoint
        checkpoint_path = cfg.eval.checkpoint_path
        log(f"Using checkpoint: {checkpoint_path}")
        log(f"Using SEED: {rseed}")
        transformers.set_seed(seed=rseed)

        model_pipeline = PatchTSTPipeline.from_pretrained(
            mode="pretrain",
            pretrain_path=checkpoint_path,
            device_map=cfg.eval.device,
            torch_dtype=getattr(torch, cfg.eval.torch_dtype, torch.float32),
        )

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
        if cfg.eval.save_completions:
            log(f"Saving completions to {completions_dict_path}")
            with open(completions_dict_path, "wb") as f:
                pickle.dump(completions_dict, f)
            log(f"Saved completions to {completions_dict_path}")

    # Compute naive interpolations
    naive_interp_str = None
    if cfg.eval.compute_naive_interpolations:
        interpolation_method = cfg.eval.naive_interpolation_method
        polynomial_degree = cfg.eval.naive_interpolation_polynomial_degree
        piecewise_spline_degree = cfg.eval.naive_interpolation_piecewise_spline_degree
        # naive_interp_str = f"polynomial{polynomial_degree}" if interpolation_method == "polynomial" else "linear"
        naive_interp_str = interpolation_method
        if interpolation_method == "piecewise_spline":
            naive_interp_str += f"_k{piecewise_spline_degree}"
        elif interpolation_method == "polynomial":
            naive_interp_str += f"_degree{polynomial_degree}"
        elif interpolation_method == "linear":
            naive_interp_str = "linear"
        else:
            raise ValueError(f"Invalid interpolation method: {interpolation_method}")
        log(f"Using {interpolation_method}")
        if interpolation_method == "polynomial":
            log(f"Using polynomial degree: {polynomial_degree}")
        log(f"naive interpolation summary string: {naive_interp_str}")

        naive_interpolations_dict_path = os.path.join(
            metrics_save_dir,
            f"{metrics_fname}_naive_{naive_interp_str}_start{start_time}_end{end_time}_rseed{rseed}.pkl",
        )
        log(f"Computing naive interpolations and saving to {naive_interpolations_dict_path}")
        naive_interpolations_dict = get_naive_interpolation(
            completions_dict,
            interpolation_method=interpolation_method,
            polynomial_degree=polynomial_degree,
        )
        with open(naive_interpolations_dict_path, "wb") as f:
            pickle.dump(naive_interpolations_dict, f)
        log(f"Saved naive interpolations to {naive_interpolations_dict_path}")

        if cfg.eval.compute_gp_dims:
            log(f"Computing GP dimensions for naive interpolations: {naive_interp_str}")
            # now create a new naive_interpolations_dict_full that contains the "groundtruth" from completions_dict, and replace the "completions" from completions_dict with the values from naive_interpolations_dict
            naive_interpolations_dict_full = {}
            for system_name in completions_dict.keys():
                naive_interpolations_dict_full[system_name] = {
                    "processed_context": completions_dict[system_name]["processed_context"],
                    "completions": naive_interpolations_dict[system_name],
                    "timestep_mask": completions_dict[system_name]["timestep_mask"],
                }

            if cfg.eval.debug_mode:
                log("Plotting example naive interpolations (debug mode enabled)")
                # plot 20 randomly chosen systems of the naive interpolations. For each plot, plot the completions, the processed_context, and the naive interpolations. Plot each dimension in a separate subplot.
                all_system_names = list(naive_interpolations_dict.keys())
                rng = np.random.default_rng(rseed)
                system_names = rng.choice(all_system_names, size=min(30, len(all_system_names)), replace=False).tolist()
                fig_save_dir = os.path.join("figures", naive_interp_str)
                os.makedirs(fig_save_dir, exist_ok=True)
                for system_name in tqdm(system_names, desc="Plotting naive interpolations"):
                    # Get the data for this system
                    processed_context_sys = naive_interpolations_dict_full[system_name]["processed_context"]
                    completions_sys = completions_dict[system_name]["completions"]
                    naive_interp_sys = naive_interpolations_dict[system_name]
                    timestep_mask_sys = naive_interpolations_dict_full[system_name]["timestep_mask"]
                    print(f"processed_context_sys shape: {processed_context_sys.shape}")
                    print(f"completions_sys shape: {completions_sys.shape}")
                    print(f"naive_interp_sys shape: {naive_interp_sys.shape}")
                    print(f"timestep_mask_sys shape: {timestep_mask_sys.shape}")

                    # Plot completions
                    plot_model_completion(
                        completions=completions_sys,
                        processed_context=processed_context_sys,
                        timestep_mask=timestep_mask_sys,
                        save_path=os.path.join(
                            fig_save_dir,
                            f"{naive_interp_str}_completions_plot_{system_name}_start{start_time}_end{end_time}.pdf",
                        ),
                    )

                    # Plot naive interpolation
                    plot_model_completion(
                        completions=naive_interp_sys,
                        processed_context=processed_context_sys,
                        timestep_mask=timestep_mask_sys,
                        save_path=os.path.join(
                            fig_save_dir,
                            f"{naive_interp_str}_naive_interp_plot_{system_name}_start{start_time}_end{end_time}.pdf",
                        ),
                    )
            del completions_dict

            log(f"Computing GP dimensions for naive interpolations: {naive_interp_str}")
            log(f"Using {cfg.eval.num_processes} processes")

            naive_metrics_save_dir = os.path.join(metrics_save_dir, naive_interp_str)
            os.makedirs(naive_metrics_save_dir, exist_ok=True)
            log(f"This script will save the GP dimensions for the naive interpolations to: {naive_metrics_save_dir}")

            gp_dims = get_gp_dims(naive_interpolations_dict_full, n_jobs=cfg.eval.num_processes)
            metrics_path = os.path.join(
                naive_metrics_save_dir,
                f"{metrics_fname}_start{start_time}_end{end_time}_rseed{rseed}_{naive_interp_str}.json",
            )

            log(f"Saving GP dimensions for naive interpolations: {naive_interp_str} to {metrics_path}")
            with open(metrics_path, "w") as f:
                json.dump(gp_dims, f, indent=4)
            log(f"Saved GP dimensions for naive interpolations: {naive_interp_str} to {metrics_path}")
        else:
            log(
                f"Skipping GP dimensions computation for naive interpolations, only saved in naive_interpolations_dict, to {naive_interpolations_dict_path}"
            )

    else:
        # Compute GP dimensions
        if cfg.eval.compute_gp_dims:
            log("Computing GP dimensions")
            log(f"Using {cfg.eval.num_processes} processes")

            gp_dims = get_gp_dims(completions_dict, n_jobs=cfg.eval.num_processes)

            metrics_path = os.path.join(
                metrics_save_dir, f"{metrics_fname}_start{start_time}_end{end_time}_rseed{rseed}.json"
            )

            log(f"Saving GP dimensions to {metrics_path}")
            with open(metrics_path, "w") as f:
                json.dump(gp_dims, f, indent=4)
        else:
            log(f"Skipping GP dimensions computation, only saved completions, to {completions_dict_path}")


if __name__ == "__main__":
    main()
