"""
Analyze a dataset of pre-computed trajectories, loading from either Arrow files or npy files
"""

import argparse
import os

# Using itertools.chain
from itertools import chain
from multiprocessing import Pool
from typing import Callable, Dict

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


def plot_distribution_from_ensemble(
    ensemble: Dict[str, np.ndarray], plot_title: str, save_dir: str
) -> None:
    vals = list(chain(*ensemble.values()))
    plot_name = plot_title.lower().replace(" ", "_")

    np_save_path = os.path.join(save_dir, f"{plot_name}.npy")
    np.save(np_save_path, vals)
    print(f"Saved npy file to {np_save_path}")

    plot_save_path = os.path.join(save_dir, f"{plot_name}.png")
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


def plot_distribution_from_npy(npy_path: str, plot_title: str, save_dir: str) -> None:
    vals = np.load(npy_path)
    plot_name = plot_title.lower().replace(" ", "_")
    plot_save_path = os.path.join(save_dir, f"{plot_name}.png")
    plt.hist(
        vals,
        bins=100,
        density=True,
        color="tab:blue",
    )
    plt.title(plot_title)
    plt.ylabel("Density")
    plt.yscale("log")
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
            bins=bins,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="debug")
    parser.add_argument("--save_dir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ensemble = make_ensemble_from_arrow_dir(DATA_DIR, args.split, one_dim_target=False)
    print(ensemble.keys())
    lyapunov_exponents = compute_quantities_multiprocessed(
        ensemble, compute_fn=compute_lyapunov_exponents
    )
    plot_distribution_from_ensemble(
        lyapunov_exponents,
        plot_title="Max Lyapunov Exponents",
        save_dir=args.save_dir,
    )

    # plot_distribution_from_npy(
    #     npy_path=os.path.join(args.save_dir, "max_lyapunov_exponents.npy"),
    #     plot_title="Max Lyapunov Exponents",
    #     save_dir=args.save_dir,
    # )

    # plot_all_distributions(
    #     npy_paths={
    #         "train": os.path.join(args.save_dir, "max_lyapunov_exponents_train.npy"),
    #         "test": os.path.join(args.save_dir, "max_lyapunov_exponents_test.npy"),
    #         "skew flow": os.path.join(
    #             args.save_dir, "max_lyapunov_exponents_skew_flow.npy"
    #         ),
    #         # "skew_phase": os.path.join(
    #         #     args.save_dir, "max_lyapunov_exponents_skew_phase.npy"
    #         # ),
    #     },
    #     plot_title="Max Lyapunov Exponents",
    #     save_name="all_max_lyapunov_exponents",
    #     save_dir=args.save_dir,
    # )
