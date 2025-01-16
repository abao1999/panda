"""
Analyze a dataset of pre-computed trajectories, loading from either Arrow files or npy files
"""

import json
import logging
import os
from functools import partial
from itertools import chain
from multiprocessing import Pool
from typing import Callable, Dict, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
from dysts.analysis import max_lyapunov_exponent_rosenstein

import dystformer.attractor as attractor
from dystformer.attractor import AttractorValidator
from dystformer.utils import make_ensemble_from_arrow_dir


def standardize_trajectories(trajectories: np.ndarray) -> np.ndarray:
    """
    Standardize the trajectories in a safe way to avoid NaNs
    trajectories: shape (n_samples, n_dims, timesteps) or (n_samples, n_dims)
    """
    if trajectories.ndim == 3:
        mean = np.nanmean(trajectories, axis=-1)[:, :, None]
        std = np.nanstd(trajectories, axis=-1)[:, :, None]
    elif trajectories.ndim == 2:
        mean = np.nanmean(trajectories, axis=-1)[:, None]
        std = np.nanstd(trajectories, axis=-1)[:, None]
    else:
        raise ValueError(
            f"Expected trajectories to be shape (n_samples, n_dims, timesteps) or (n_samples, n_dims), got {trajectories.shape}"
        )
    std = np.where(std < 1e-10, 1e-10, std)
    return (trajectories - mean) / std


def compute_lyapunov_exponents(
    dyst_name: str,
    trajectories: np.ndarray,
    trajectory_len: int = 1024,
    transient_time_frac: float = 0.05,
) -> np.ndarray:
    """
    Compute the Lyapunov exponents for a specified system.
    """
    lyapunov_exponents = []
    for traj in trajectories:
        traj = traj[:, :trajectory_len]
        spectrum = [
            max_lyapunov_exponent_rosenstein(traj.T, trajectory_len=trajectory_len)
        ]
        lyapunov_exponents.extend(spectrum)
    return np.array(lyapunov_exponents)


def compute_quantile_limits(
    dyst_name: str,
    trajectories: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Comutes the high and low values for the instance normalized trajectories
    """
    assert trajectories.ndim == 3, (
        "expected trajectories to be shape (n_samples, n_dims, timesteps)"
    )
    standard_traj = standardize_trajectories(trajectories)
    high = np.max(standard_traj, axis=(1, 2))
    low = np.min(standard_traj, axis=(1, 2))
    return high, low


def multiprocessed_compute_wrapper(
    ensemble: Dict[str, np.ndarray],
    compute_fn: Callable,
) -> Dict[str, np.ndarray]:
    """
    Compute the quantities for the ensemble of trajectories in parallel (multiprocessed)
    Args:
        ensemble: Ensemble of trajectories
        compute_fn: Function to compute the quantity
    Returns:
        Dict[str, np.ndarray]: Dictionary of the computed quantities for each system. Key is system name, value is the computed quantity
    """
    with Pool() as pool:
        results = pool.starmap(
            compute_fn,
            [(dyst_name, all_traj) for dyst_name, all_traj in ensemble.items()],
        )
    return {k: v for k, v in zip(list(ensemble.keys()), results)}


def plot_from_npy_paths(
    npy_paths: Dict[str, str],
    plot_title: str,
    save_name: str,
    save_dir: str,
    log_scale: bool = False,
    show_legend: bool = False,
) -> None:
    """
    Plot the distribution of values in the npy files
    """
    plot_save_path = os.path.join(save_dir, f"{save_name}.png")
    all_vals = []
    for data_split, npy_path in npy_paths.items():
        vals = np.load(npy_path)
        all_vals.append(vals)

    # Determine the bins based on the combined data
    combined_vals = np.concatenate(all_vals)
    bins = np.histogram_bin_edges(combined_vals, bins=100)

    for i, (data_split, npy_path) in enumerate(npy_paths.items()):
        vals = np.load(npy_path)
        plt.hist(
            vals,
            bins=bins,  # type: ignore
            density=True,
            alpha=0.5,
            color=plt.get_cmap("tab10")(i),
            label=data_split,
        )
    plt.title(plot_title)
    plt.ylabel("Density")
    plt.grid(True)
    if log_scale:
        plt.yscale("log")
    if show_legend:
        plt.legend()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()


def plot_from_npy_paths_scatter(
    npy_paths: Dict[str, str],
    plot_title: str,
    save_name: str,
    save_dir: str,
    show_legend: bool = False,
) -> None:
    """
    Plot the scatter plot of the high and low values in the npy files
    """
    plot_save_path = os.path.join(save_dir, f"{save_name}.png")
    all_vals = []
    for data_split, npy_path in npy_paths.items():
        vals = np.load(npy_path)
        if vals.shape[-1] != 2:
            raise ValueError(f"Expected 2D data, got {vals.shape}")
        all_vals.append(vals)

    for i, (data_split, npy_path) in enumerate(npy_paths.items()):
        vals = np.load(npy_path)
        plt.scatter(
            vals[:, 0],
            vals[:, 1],
            alpha=0.2,
            color=plt.get_cmap("tab10")(i),
            label=data_split,
        )
    plt.title(plot_title)
    plt.xlabel("min")
    plt.ylabel("max")
    plt.gca().invert_xaxis() if vals[:, 0].min() < 0 else None
    plt.gca().invert_yaxis() if vals[:, 1].min() < 0 else None
    if show_legend:
        plt.legend()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    save_dir = os.path.join(cfg.analysis.save_dir, cfg.analysis.split)
    plot_save_dir = os.path.join(cfg.analysis.plots_dir, cfg.analysis.split)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)

    # make ensemble from saved trajectories in Arrow files
    ensemble = make_ensemble_from_arrow_dir(
        cfg.analysis.data_dir, cfg.analysis.split, one_dim_target=False
    )
    logger.info(
        f"Loaded {len(ensemble)} systems from {cfg.analysis.data_dir} split {cfg.analysis.split}"
    )

    if cfg.analysis.filter_ensemble:
        logger.info(
            f"Filtering ensemble with {len(cfg.analysis.attractor_tests)} tests"
        )
        attractor_tests = []
        for test_name in cfg.analysis.attractor_tests:
            save_path = os.path.join(save_dir, f"{test_name}_rejected_samples.json")
            test_kwargs = getattr(cfg.analysis, test_name)
            test_fn = getattr(attractor, test_name)
            test_fn = partial(test_fn, **test_kwargs)
            logger.info(
                f"Adding test {test_name} to validator with kwargs: \n {test_kwargs}"
            )
            attractor_tests.append(test_fn)

        validator = AttractorValidator(
            tests=attractor_tests,
            transient_time_frac=0.0,
        )
        valid_ensemble, failed_ensemble = validator.multiprocessed_filter_ensemble(
            ensemble,
        )
        failed_samples = validator.failed_samples
        logger.info(f"Failed {len(failed_samples)} samples")
        json.dump(
            failed_samples,
            open(os.path.join(save_dir, "failed_samples.json"), "w"),
            indent=4,
        )

    if cfg.analysis.compute_quantile_limits:
        quantile_limits = multiprocessed_compute_wrapper(
            ensemble, compute_fn=compute_quantile_limits
        )

        # save the quantile limits as npy files
        high_vals = np.array(list(chain(*[v[0] for v in quantile_limits.values()])))
        low_vals = np.array(list(chain(*[v[1] for v in quantile_limits.values()])))
        path_low_vals = os.path.join(save_dir, "quantile_low.npy")
        path_high_vals = os.path.join(save_dir, "quantile_high.npy")
        np.save(path_low_vals, low_vals)
        np.save(path_high_vals, high_vals)

        all_vals = np.dstack((low_vals, high_vals)).squeeze()
        path_all_vals = os.path.join(save_dir, "quantile_low_high.npy")
        np.save(path_all_vals, all_vals)

        # plot the quantile limits
        plot_from_npy_paths(
            npy_paths={
                "min": path_low_vals,
                "max": path_high_vals,
            },
            plot_title="Quantile Limits",
            save_name="quantlim",
            save_dir=plot_save_dir,
            log_scale=False,
        )

        plot_from_npy_paths_scatter(
            npy_paths={
                "quantile_low_high": path_all_vals,
            },
            plot_title="Quantile Limits",
            save_name="quantlim_scatter",
            save_dir=plot_save_dir,
            show_legend=False,
        )

    if cfg.analysis.compute_max_lyapunov_exponents:
        save_path = os.path.join(save_dir, "max_lyapunov_exponents.npy")
        max_lyapunov_exponents = multiprocessed_compute_wrapper(
            ensemble, compute_fn=compute_lyapunov_exponents
        )
        max_lyapunov_exponents_vals = np.array(
            list(chain(*max_lyapunov_exponents.values()))
        )

        np.save(save_path, max_lyapunov_exponents_vals)

        # plot the max lyapunov exponents
        plot_from_npy_paths(
            npy_paths={
                "max_lyapunov_exponents": save_path,
            },
            plot_title="Max Lyapunov Exponents",
            save_name="max_lyap",
            save_dir=plot_save_dir,
            show_legend=False,
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
