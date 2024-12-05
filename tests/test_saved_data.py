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
    plot_trajs_multivariate,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def plot_saved_data(
    dyst_names_lst: List[str],
    split: str,
    one_dim_target: bool = False,
    samples_subset_dict: Optional[Dict[str, List[int]]] = None,
    plot_name_suffix: Optional[str] = None,
    plot_save_dir: str = "tests/figs",
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

        filepaths = get_system_filepaths(dyst_name, DATA_DIR, split)
        print(f"{dyst_name} filepaths: ", filepaths)

        dyst_coords_samples = accumulate_coords(filepaths, one_dim_target)
        # # cut off only first 512 time steps
        # dyst_coords_samples = dyst_coords_samples[:, :, :512]

        # plot the trajectories
        plot_name = f"{dyst_name}_{plot_name_suffix}" if plot_name_suffix else dyst_name
        plot_trajs_multivariate(
            dyst_coords_samples,
            save_dir=plot_save_dir,
            plot_name=plot_name,
            samples_subset=samples_subset,
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
    args = parser.parse_args()

    if args.split is None:
        raise ValueError("Split must be provided for loading data")

    if args.dysts_names == ["all"]:
        split_dir = os.path.join(DATA_DIR, args.split)
        dyst_names_lst = [
            folder.name for folder in Path(split_dir).iterdir() if folder.is_dir()
        ]
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
        split=args.split,
        one_dim_target=args.one_dim_target,
        samples_subset_dict=samples_subset_dict,
        plot_name_suffix="failures"
        if args.samples_subset == "failed_samples"
        else None,
        plot_save_dir=args.plot_save_dir,
    )
