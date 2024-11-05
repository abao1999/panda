"""
Analyze a dataset of pre-computed trajectories, loading from either Arrow files or npy files
"""

import argparse
import os
from itertools import chain

# Using itertools.chain
from multiprocessing import Pool
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from dysts.analysis import max_lyapunov_exponent_rosenstein

from dystformer.utils import make_ensemble_from_arrow_dir

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def compute_lyapunov_exponents(
    dyst_name: str,
    all_traj: np.ndarray,
) -> np.ndarray:
    """
    Compute the Lyapunov exponents for a specified system.
    TODO: currently only computes the max Lyapunov exponent, but could be extended to compute the full spectrum
    Args:
        dyst_name: Name of the dynamical system.
        all_traj: All trajectories for the specified system.
    Returns:
        np.ndarray: Lyapunov exponents for the specified system.
    """
    lyapunov_exponents = []
    print(
        f"Computing Lyapunov exponents for {dyst_name} samples with shape {all_traj.shape}"
    )
    for traj in all_traj:
        spectrum = [max_lyapunov_exponent_rosenstein(traj.T, trajectory_len=200)]
        lyapunov_exponents.extend(spectrum)
    return np.array(lyapunov_exponents)


def compute_quantile_limits(
    dyst_name: str,
    all_traj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the high and low values for the instance normalized trajectories
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
    with Pool() as pool:
        results = pool.starmap(
            compute_fn,
            [(dyst_name, all_traj) for dyst_name, all_traj in ensemble.items()],
        )
    return {k: v for k, v in zip(list(ensemble.keys()), results)}


def plot_distribution(
    vals: np.ndarray, plot_title: str, save_name: str, save_dir: str
) -> None:
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
    npy_paths: Dict[str, str], plot_title: str, save_name: str, save_dir: str
) -> None:
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
    plt.yscale("log")
    plt.legend()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_save_path}")


def plot_all_distributions_scatter(
    npy_paths: Dict[str, str], plot_title: str, save_name: str, save_dir: str
) -> None:
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
    parser.add_argument("--save_dir", type=str, default="plots")
    parser.add_argument("--suffix_name", type=str, default="")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    plot_all_distributions_scatter(
        npy_paths={
            "train": os.path.join(args.save_dir, "quantile_limits_train.npy"),
            "skew flow": os.path.join(args.save_dir, "quantile_limits_skew_flow.npy"),
        },
        plot_title="Quantile Limits",
        save_name="quantile_limits_scatter",
        save_dir=args.save_dir,
    )

    plot_all_distributions(
        npy_paths={
            "high": os.path.join(args.save_dir, "quantile_limits_train_high.npy"),
            "low": os.path.join(args.save_dir, "quantile_limits_train_low.npy"),
        },
        plot_title="Quantile Limits",
        save_name="quantile_limits_train",
        save_dir=args.save_dir,
    )

    plot_all_distributions(
        npy_paths={
            "high": os.path.join(args.save_dir, "quantile_limits_skew_flow_high.npy"),
            "low": os.path.join(args.save_dir, "quantile_limits_skew_flow_low.npy"),
        },
        plot_title="Quantile Limits",
        save_name="quantile_limits_skew_flow",
        save_dir=args.save_dir,
    )
    exit()

    suffix_name = f"_{args.suffix_name}" if args.suffix_name else ""
    # save_name = f"max_lyapunov_exponents{suffix_name}"
    save_name = f"quantile_limits{suffix_name}"

    # make ensemble from saved trajectories in Arrow files
    ensemble = make_ensemble_from_arrow_dir(DATA_DIR, args.split, one_dim_target=False)
    print(ensemble.keys())

    # compute quantity of interest
    ## Max Lyapunov Exponents
    # lyapunov_exponents = compute_quantities_multiprocessed(
    #     ensemble, compute_fn=compute_lyapunov_exponents
    # )
    # le_vals = np.array(list(chain(*ensemble.values())))

    ## Quantile Limits
    quantile_limits = compute_quantities_multiprocessed(
        ensemble, compute_fn=compute_quantile_limits
    )
    print(quantile_limits.keys())
    print(quantile_limits.values())

    high_vals = np.array(list(chain(*[v[0] for v in quantile_limits.values()])))
    low_vals = np.array(list(chain(*[v[1] for v in quantile_limits.values()])))
    np.save(os.path.join(args.save_dir, f"{save_name}_high.npy"), high_vals)
    np.save(os.path.join(args.save_dir, f"{save_name}_low.npy"), low_vals)
    low_high_vals = np.dstack((low_vals, high_vals)).squeeze()
    np.save(os.path.join(args.save_dir, f"{save_name}.npy"), low_high_vals)

    plot_all_distributions_scatter(
        npy_paths={
            suffix_name.replace("_", " "): os.path.join(
                args.save_dir, f"{save_name}.npy"
            ),
        },
        plot_title="Quantile Limits",
        save_name=f"{save_name}_scatter",
        save_dir=args.save_dir,
    )

    # save npy file of computed values
    # np_save_path = os.path.join(args.save_dir, f"{save_name}.npy")
    # print(f"Saving npy file to {np_save_path}")
    # np.save(os.path.join(args.save_dir, f"{save_name}.npy"), le_vals)

    # plot_all_distributions(
    #     npy_paths={
    #         "high": os.path.join(args.save_dir, f"{save_name}_high.npy"),
    #         "low": os.path.join(args.save_dir, f"{save_name}_low.npy"),
    #     },
    #     plot_title="Quantile Limits",
    #     save_name=save_name,
    #     save_dir=args.save_dir,
    # )

    # plot_all_distributions(
    #     npy_paths={
    #         "train": os.path.join(args.save_dir, "max_lyapunov_exponents_train.npy"),
    #         "test": os.path.join(args.save_dir, "max_lyapunov_exponents_test.npy"),
    #         "skew flow": os.path.join(
    #             args.save_dir, "max_lyapunov_exponents_skew_flow.npy"
    #         ),
    #     },
    #     plot_title="Max Lyapunov Exponents",
    #     save_name="all_max_lyapunov_exponents",
    #     save_dir=args.save_dir,
    # )
