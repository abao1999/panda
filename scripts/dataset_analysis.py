"""
Analyze a dataset of pre-computed trajectories, loading from either Arrow files or npy files
"""

import argparse
import os
from itertools import chain
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from dysts.analysis import max_lyapunov_exponent_rosenstein

from dystformer.utils import make_ensemble_from_arrow_dir

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def compute_lyapunov_exponents(
    dyst_name: str,
    all_traj: np.ndarray,
    trajectory_len: int = 200,
) -> np.ndarray:
    """
    Compute the Lyapunov exponents for a specified system.
    Args:
        dyst_name: Name of the dynamical system.
        all_traj: All trajectories for the specified system.
        trajectory_len: Length of the trajectory to use for Lyapunov exponent computation
    Returns:
        np.ndarray: Lyapunov exponents for the specified system.
    """
    lyapunov_exponents = []
    print(
        f"Computing Lyapunov exponents for {dyst_name} samples with shape {all_traj.shape}"
    )
    for traj in all_traj:
        spectrum = [
            max_lyapunov_exponent_rosenstein(traj.T, trajectory_len=trajectory_len)
        ]
        lyapunov_exponents.extend(spectrum)
    return np.array(lyapunov_exponents)


def compute_quantile_limits(
    dyst_name: str,
    all_traj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the high and low values for the instance normalized trajectories
    Args:
        dyst_name: Name of the dynamical system (not used)
        all_traj: All trajectories for the specified system
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the high and low values for the trajectories
    """
    high_vals = []
    low_vals = []
    for traj in all_traj:
        standard_traj = (traj - np.mean(traj, axis=-1)[:, None]) / np.std(
            traj, axis=-1
        )[:, None]
        high = np.max(standard_traj)
        low = np.min(standard_traj)
        high_vals.append(high)
        low_vals.append(low)
    return np.array(high_vals), np.array(low_vals)


def compute_quantities_multiprocessed(
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


def filter_quantile_limits(
    dyst_name: str,
    all_traj: np.ndarray,
    max_abs_val: float = 15,
) -> List[int]:
    """
    Filter the trajectories for the given system based on the quantile limits
    Args:
        dyst_name: Name of the dynamical system (not used)
        all_traj: All trajectories for the specified system
        max_abs_val: Maximum absolute value for the whitened trajectory
    Returns:
        List[int]: List of the invalid trajectory sample indices
    """
    samples_to_remove = []
    for sample_idx, traj in enumerate(all_traj):
        standard_traj = (traj - np.mean(traj, axis=-1)[:, None]) / np.std(
            traj, axis=-1
        )[:, None]
        if np.max(np.abs(standard_traj)) > max_abs_val:
            samples_to_remove.append(sample_idx)
    return samples_to_remove


def filter_saved_trajectories_multiprocessed(
    ensemble: Dict[str, np.ndarray],
    filter_fn: Callable,
) -> Dict[str, np.ndarray]:
    """
    Filter the saved trajectories for the ensemble in parallel (multiprocessed). Simply saves rejected samples into dict
    Args:
        ensemble: Ensemble of trajectories
        filter_fn: Function to return a list of the invalid trajectory sample indices for each system
    Returns:
        Dict[str, List[int]]: Dictionary of rejected samples for each system. Key is system name, value is list of rejected sample indices
    """
    with Pool() as pool:
        results = pool.starmap(
            filter_fn,
            [(dyst_name, all_traj) for dyst_name, all_traj in ensemble.items()],
        )
    rejected_samples = {k: v for k, v in zip(list(ensemble.keys()), results) if v}
    return rejected_samples


def plot_distribution(
    vals: np.ndarray, plot_title: str, save_name: str, save_dir: str
) -> None:
    """
    Plot the distribution of values in the npy file
    """
    plot_save_path = os.path.join(save_dir, f"{save_name}.png")
    plt.hist(
        vals,
        bins=100,
        density=True,
        color="tab:blue",
    )
    plt.title(plot_title)
    plt.ylabel("Density")
    # plt.yscale("log")
    plt.grid(True)
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_save_path}")


def plot_all_distributions(
    npy_paths: Dict[str, str],
    plot_title: str,
    save_name: str,
    save_dir: str,
    log_scale: bool = True,
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
    plt.legend()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_save_path}")


def plot_all_distributions_scatter(
    npy_paths: Dict[str, str], plot_title: str, save_name: str, save_dir: str
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
        print(vals.shape)
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
    plt.legend()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="debug")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--plots_dir", type=str, default="plots")
    parser.add_argument("--suffix_name", type=str, default="")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # plot_all_distributions(
    #     npy_paths={
    #         "skew flow": os.path.join(
    #             args.save_dir, "max_lyapunov_exponents_big_flow_skew.npy"
    #         ),
    #         "train": os.path.join(
    #             args.save_dir, "max_lyapunov_exponents_signed_perts_train.npy"
    #         ),
    #     },
    #     plot_title="Max Lyapunov Exponents",
    #     save_name="max_lyapunov_exponents_all",
    #     save_dir=args.plots_dir,
    #     log_scale=False,
    # )
    # exit()

    # # plot_all_distributions_scatter(
    # #     npy_paths={
    # #         "train": os.path.join(args.save_dir, "quantile_limits_train.npy"),
    # #         "skew flow": os.path.join(args.save_dir, "quantile_limits_skew_flow.npy"),
    # #     },
    # #     plot_title="Quantile Limits",
    # #     save_name="quantile_limits_scatter",
    # #     save_dir=args.save_dir,
    # # )
    # exit()

    suffix_name = f"_{args.suffix_name}" if args.suffix_name else ""
    # save_name = f"max_lyapunov_exponents{suffix_name}"
    save_name = f"quantile_limits{suffix_name}"

    # make ensemble from saved trajectories in Arrow files
    ensemble = make_ensemble_from_arrow_dir(DATA_DIR, args.split, one_dim_target=False)

    # ## Filter saved trajectories
    # import json
    # from functools import partial

    # rejected_samples = filter_saved_trajectories_multiprocessed(
    #     ensemble, filter_fn=partial(filter_quantile_limits, max_abs_val=15)
    # )
    # # Write the rejected samples to a JSON file
    # rejected_samples_json = os.path.join(
    #     args.save_dir, f"rejected_samples{suffix_name}.json"
    # )
    # with open(rejected_samples_json, "w") as f:
    #     json.dump(rejected_samples, f, indent=4)
    # print(f"Saved rejected samples to {rejected_samples_json}")
    # print(f"Rejected {len(rejected_samples)} samples")

    # exit()

    ## Quantile Limits
    quantile_limits = compute_quantities_multiprocessed(
        ensemble, compute_fn=compute_quantile_limits
    )

    high_vals = np.array(list(chain(*[v[0] for v in quantile_limits.values()])))
    low_vals = np.array(list(chain(*[v[1] for v in quantile_limits.values()])))
    np.save(os.path.join(args.save_dir, f"{save_name}_high.npy"), high_vals)
    np.save(os.path.join(args.save_dir, f"{save_name}_low.npy"), low_vals)

    plot_all_distributions(
        npy_paths={
            "low": os.path.join(args.save_dir, f"{save_name}_low.npy"),
            "high": os.path.join(args.save_dir, f"{save_name}_high.npy"),
        },
        plot_title="Quantile Limits",
        save_name=f"{save_name}_hist",
        save_dir=args.plots_dir,
    )

    all_vals = np.dstack((low_vals, high_vals)).squeeze()
    np.save(os.path.join(args.save_dir, f"{save_name}.npy"), all_vals)

    plot_all_distributions_scatter(
        npy_paths={
            suffix_name.replace("_", " "): os.path.join(
                args.save_dir, f"{save_name}.npy"
            ),
        },
        plot_title="Quantile Limits",
        save_name=f"{save_name}_scatter",
        save_dir=args.plots_dir,
    )

    # # Max Lyapunov Exponents
    # max_lyapunov_exponents = compute_quantities_multiprocessed(
    #     ensemble, compute_fn=compute_lyapunov_exponents
    # )
    # max_lyapunov_exponents_vals = np.array(
    #     list(chain(*[v for v in max_lyapunov_exponents.values()]))
    # )
    # # max_lyapunov_exponents_vals = np.array(list(chain(*max_lyapunov_exponents.values())))

    # np.save(
    #     os.path.join(args.save_dir, f"max_lyapunov_exponents{suffix_name}.npy"),
    #     max_lyapunov_exponents_vals,
    # )
    # plot_all_distributions(
    #     npy_paths={
    #         suffix_name.replace("_", " "): os.path.join(
    #             args.save_dir, f"max_lyapunov_exponents{suffix_name}.npy"
    #         ),
    #     },
    #     plot_title="Max Lyapunov Exponents",
    #     save_name=f"max_lyapunov_exponents{suffix_name}_hist",
    #     save_dir=args.plots_dir,
    # )
