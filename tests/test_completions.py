"""
Load saved Arrow data files and plot the trajectories.
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Optional

from dystformer.utils import (
    accumulate_coords,
    get_system_filepaths,
    plot_completions_evaluation,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def plot_saved_data(
    dyst_names_lst: List[str],
    split_completions: str,
    split_context: str,
    split_mask: str,
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

        # completions
        filepaths_completions = get_system_filepaths(
            dyst_name, DATA_DIR, split_completions
        )
        dyst_coords_samples_completions = accumulate_coords(
            filepaths_completions, one_dim_target, num_samples=num_samples_plot
        )

        # context
        filepaths_context = get_system_filepaths(dyst_name, DATA_DIR, split_context)
        dyst_coords_samples_context = accumulate_coords(
            filepaths_context, one_dim_target, num_samples=num_samples_plot
        )

        # mask
        filepaths_mask = get_system_filepaths(dyst_name, DATA_DIR, split_mask)
        dyst_mask_samples = accumulate_coords(
            filepaths_mask, one_dim_target, num_samples=num_samples_plot
        )

        # plot the trajectories
        plot_name = f"{dyst_name}_{plot_name_suffix}" if plot_name_suffix else dyst_name
        plot_completions_evaluation(
            completions=dyst_coords_samples_completions,
            context=dyst_coords_samples_context,
            mask=dyst_mask_samples,
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
        "--split_completions",
        help="Split of the completions",
        type=str,
        default="eval/completions",
    )
    parser.add_argument(
        "--split_context",
        help="Split of the context",
        type=str,
        default="eval/patch_input",
    )
    parser.add_argument(
        "--split_mask",
        help="Split of the context",
        type=str,
        default="eval/timestep_masks",
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
        default="figs/completions",
    )
    args = parser.parse_args()

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
        split_completions=args.split_completions,
        split_context=args.split_context,
        split_mask=args.split_mask,
        one_dim_target=args.one_dim_target,
        samples_subset_dict=samples_subset_dict,
        plot_name_suffix="failures"
        if args.samples_subset == "failed_samples"
        else None,
        plot_save_dir=args.plot_save_dir,
    )
