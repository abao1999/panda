"""
Load saved Arrow data files and plot the trajectories.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from dystformer.utils import (
    accumulate_coords,
    get_system_filepaths,
    plot_forecast_evaluation,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def plot_saved_data(
    dyst_names_lst: List[str],
    split_forecasts: str,
    split_ground_truth: str,
    context_length: int,
    one_dim_target: bool = False,
    samples_subset_dict: Optional[Dict[str, List[int]]] = None,
    plot_name_suffix: Optional[str] = None,
    plot_save_dir: str = "tests/figs",
    num_samples_plot: int = 6,
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

        filepaths_forecasts = get_system_filepaths(dyst_name, DATA_DIR, split_forecasts)
        filepaths_ground_truth = get_system_filepaths(
            dyst_name, DATA_DIR, split_ground_truth
        )
        dyst_coords_samples_forecasts = accumulate_coords(
            filepaths_forecasts, one_dim_target, num_samples=num_samples_plot
        )
        dyst_coords_samples_ground_truth = accumulate_coords(
            filepaths_ground_truth, one_dim_target, num_samples=num_samples_plot
        )
        # plot the trajectories
        plot_name = f"{dyst_name}_{plot_name_suffix}" if plot_name_suffix else dyst_name
        plot_forecast_evaluation(
            forecasts=dyst_coords_samples_forecasts,
            ground_truth=dyst_coords_samples_ground_truth,
            context_length=context_length,
            save_dir=plot_save_dir,
            plot_name=plot_name,
            samples_subset=samples_subset,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dysts_names", help="Names of the dynamical systems", nargs="+", type=str
    )
    parser.add_argument(
        "--split_forecasts",
        help="Split of the forecasts",
        type=str,
        default="eval/forecasts",
    )
    parser.add_argument(
        "--split_ground_truth",
        help="Split of the ground truth",
        type=str,
        default="eval/labels",
    )
    parser.add_argument(
        "--context_length",
        help="Context length",
        type=int,
        default=512,
    )
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
        default="figs/forecasts",
    )
    parser.add_argument(
        "--num_systems",
        help="Number of systems to plot",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if args.dysts_names == ["all"]:
        dyst_names_lst = [
            d.name
            for d in Path(os.path.join(DATA_DIR, args.split_forecasts)).iterdir()
            if d.is_dir()
        ][: args.num_systems]
        breakpoint()
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

    plot_saved_data(
        dyst_names_lst,
        split_forecasts=args.split_forecasts,
        split_ground_truth=args.split_ground_truth,
        context_length=args.context_length,
        one_dim_target=args.one_dim_target,
        samples_subset_dict=samples_subset_dict,
        plot_name_suffix="failures"
        if args.samples_subset == "failed_samples"
        else None,
        plot_save_dir=args.plot_save_dir,
    )
