"""
Load saved Arrow data files and plot the trajectories.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import dysts.flows as flows
import numpy as np

from dystformer.utils import (
    accumulate_coords,
    get_system_filepaths,
    make_ensemble_from_arrow_dir,
    plot_grid_trajs_multivariate,
    plot_trajs_multivariate,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def plot_saved_data(
    dyst_names_lst: List[str],
    split: str,
    one_dim_target: bool = False,
    samples_subset_dict: Optional[Dict[str, List[int]]] = None,
    max_samples: int = 1,
    plot_default_sample: bool = True,
    plot_name_suffix: Optional[str] = None,
    plot_save_dir: str = "tests/figs",
    standardize: bool = True,
) -> None:
    """
    Plot saved Arrow data files.
    """
    for dyst_name in dyst_names_lst:
        samples_subset = None  # default to plotting all samples sequentially
        if samples_subset_dict is not None:
            if dyst_name not in samples_subset_dict:
                warnings.warn(
                    f"No samples subset found for {dyst_name}, plotting all samples sequentially"
                )
            else:
                samples_subset = samples_subset_dict[dyst_name]
                print(f"Plotting samples subset {samples_subset} for {dyst_name}")

        start_sample_idx = 0 if plot_default_sample else 1
        end_sample_idx = max_samples if plot_default_sample else max_samples + 1

        filepaths = get_system_filepaths(dyst_name, DATA_DIR, split)[
            start_sample_idx:end_sample_idx
        ]
        print(f"{dyst_name} filepaths: ", filepaths)

        dyst_coords_samples = accumulate_coords(filepaths, one_dim_target)
        coords_dim = dyst_coords_samples.shape[1]

        # plot the trajectories
        plot_name = f"{dyst_name}_{plot_name_suffix}" if plot_name_suffix else dyst_name

        is_skew = "_" in dyst_name
        if is_skew and coords_dim >= 6:  # hacky check
            driver_name, _ = dyst_name.split("_")
            driver_dim = getattr(flows, driver_name)().dimension
            driver_coords = dyst_coords_samples[:, :driver_dim, :]
            response_coords = dyst_coords_samples[:, driver_dim:, :]
            for name, coords in [
                ("driver", driver_coords),
                ("response", response_coords),
            ]:
                plot_trajs_multivariate(
                    coords,
                    save_dir=plot_save_dir,
                    plot_name=f"{plot_name}_{name}",
                    samples_subset=samples_subset,
                    standardize=standardize,
                    plot_2d_slice=False,
                    plot_projections=True,
                )
        else:
            plot_trajs_multivariate(
                dyst_coords_samples,
                save_dir=plot_save_dir,
                plot_name=plot_name,
                samples_subset=samples_subset,
                standardize=standardize,
                plot_2d_slice=False,
                plot_projections=True,
            )


def make_response_ensemble(ensemble: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Make an ensemble of just the response coordinates for each system in ensemble.
    Use case: when saving the concatenated driver and response coordinates of skew system.
    """
    driver_dims = {
        sys: getattr(flows, sys.split("_")[0])().dimension for sys in ensemble.keys()
    }
    print("got driver dims")
    response_ensemble = {
        sys: ensemble[sys][:, driver_dims[sys] :, :] for sys in ensemble.keys()
    }
    return response_ensemble


def plot_saved_data_grid(
    dyst_names_lst: List[str],
    split: str,
    max_samples: int = 6,
    plot_save_dir: str = "tests/figs",
    plot_name_suffix: Optional[str] = None,
    subplot_size: tuple[int, int] = (3, 3),
    standardize: bool = True,
) -> None:
    """
    Plot a grid of multiple systems' multivariate timeseries from dyst_data
    """
    ensemble = make_ensemble_from_arrow_dir(
        DATA_DIR, split, dyst_names_lst=dyst_names_lst
    )
    n_systems = len(ensemble)
    default_name = f"{n_systems}_systems"

    plot_name = (
        f"{default_name}_{plot_name_suffix}" if plot_name_suffix else default_name
    )
    save_path = os.path.join(plot_save_dir, f"{plot_name}.pdf")
    plot_grid_trajs_multivariate(
        ensemble,
        save_path=save_path,
        max_samples=max_samples,
        standardize=standardize,
        subplot_size=subplot_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dysts_names", help="Names of the dynamical systems", nargs="+", type=str
    )
    parser.add_argument("--split", help="Split of the data", type=str, default=None)
    parser.add_argument(
        "--one_dim_target", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--metadata_path",
        help="Path to metadata json file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--samples_subset",
        help="samples subset",
        type=str,
        choices=["failed_samples", "valid_samples"],
        default=None,
    )
    parser.add_argument(
        "--plot_save_dir",
        help="Directory to save plots",
        type=str,
        default="tests/figs",
    )
    parser.add_argument(
        "--n_systems_plot",
        help="Number of systems to plot",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--n_samples_plot",
        help="Number of samples to plot",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--skip_default_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--plot_grid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rseed",
        help="Random seed for reproducibility",
        type=int,
        default=99,
    )
    args = parser.parse_args()

    if args.split is None:
        raise ValueError("Split must be provided for loading data")

    if args.dysts_names == ["all"]:
        # choose random 9 systems to plot, using reporducible rseed
        rseed = args.rseed
        rng = np.random.default_rng(rseed)
        split_dir = os.path.join(DATA_DIR, args.split)
        dyst_names_lst = [
            folder.name for folder in Path(split_dir).iterdir() if folder.is_dir()
        ]
        dyst_names_lst = dyst_names_lst[: args.n_systems_plot]
    else:
        dyst_names_lst = args.dysts_names

    print(f"dyst names: {dyst_names_lst}")

    samples_subset_dict = None  # default to plotting all samples sequentially
    if args.metadata_path is not None and args.samples_subset is not None:
        metadata = json.load(open(args.metadata_path, "r"))
        if args.samples_subset not in metadata:
            raise ValueError(
                f"Samples subset {args.samples_subset} not found in metadata file {args.metadata_path}"
            )
        samples_subset_dict = metadata[args.samples_subset]

    plot_name_suffix = "_".join(args.split.split("/"))
    plot_name_suffix += "_failures" if args.samples_subset == "failed_samples" else ""
    if args.plot_grid:
        n_rows = round(1 + args.n_systems_plot**0.5)
        subplot_size = (n_rows, n_rows)
        plot_saved_data_grid(
            dyst_names_lst,
            split=args.split,
            max_samples=args.n_samples_plot,
            plot_name_suffix=plot_name_suffix,
            plot_save_dir=args.plot_save_dir,
            subplot_size=subplot_size,
            standardize=args.standardize,
        )
    else:
        plot_saved_data(
            dyst_names_lst,
            split=args.split,
            one_dim_target=args.one_dim_target,
            samples_subset_dict=samples_subset_dict,
            max_samples=args.n_samples_plot,
            plot_default_sample=not args.skip_default_sample,
            plot_name_suffix=plot_name_suffix,
            plot_save_dir=args.plot_save_dir,
            standardize=args.standardize,
        )
