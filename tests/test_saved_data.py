"""
Load saved Arrow data files and plot the trajectories.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from gluonts.dataset.common import FileDataset

from dystformer.utils import (
    plot_trajs_multivariate,
    plot_trajs_univariate,
    stack_and_extract_metadata,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def get_dyst_filepaths(dyst_name: str, split: str = "train") -> List[Path]:
    """
    Get filepaths for all .arrow files in data/{split}/{dyst_name}
    """
    dyst_dir = os.path.join(WORK_DIR, f"data/{split}", dyst_name)
    if not os.path.exists(dyst_dir):
        raise FileNotFoundError(f"Directory {dyst_dir} does not exist in data/{split}.")

    print(f"Found dyst directory: {dyst_dir}")
    filepaths = sorted(
        list(Path(dyst_dir).glob("*.arrow")), key=lambda x: int(x.stem.split("_")[0])
    )
    print(f"Found {len(filepaths)} files in {dyst_dir}")
    return filepaths


def plot_saved_data(
    dyst_names_lst: List[str],
    split: str,
    one_dim_target: bool = False,
    plot_univariate: bool = False,
    samples_subset_dict: Optional[Dict[str, List[int]]] = None,
    plot_name_suffix: Optional[str] = None,
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

        filepaths = get_dyst_filepaths(dyst_name, split)
        print(f"{dyst_name} filepaths: ", filepaths)

        # NOTE: this is same as accumulate_dyst_samples in tests/test_augmentations.py
        dyst_coords_samples = []
        for filepath in filepaths:
            # create dataset by reading directly from filepath into FileDataset
            gts_dataset = FileDataset(
                path=Path(filepath),
                freq="h",
                one_dim_target=one_dim_target,
            )  # TODO: consider other frequencies?

            # extract the coordinates
            dyst_coords, metadata = stack_and_extract_metadata(
                gts_dataset,
            )

            dyst_coords_samples.append(dyst_coords)

            print("data shape: ", dyst_coords.shape)
            print("metadata: ", metadata)
            print("IC: ", dyst_coords[:, 0])

        dyst_coords_samples = np.array(dyst_coords_samples)  # type: ignore
        print(dyst_coords_samples.shape)

        # plot the trajectories
        plot_name = f"{dyst_name}_{plot_name_suffix}" if plot_name_suffix else dyst_name
        plot_trajs_multivariate(
            dyst_coords_samples,
            save_dir="tests/figs",
            plot_name=plot_name,
            samples_subset=samples_subset,
        )

        if plot_univariate:
            plot_trajs_univariate(
                dyst_coords_samples,
                selected_dim=None,  # plot all dimensions
                save_dir="tests/figs/univariate",
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
        "--plot_univariate", action=argparse.BooleanOptionalAction, default=False
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
    args = parser.parse_args()

    if args.dysts_names == ["all"]:
        # get all folder names in DATA_DIR/{split}
        if args.split is None:
            raise ValueError("Split must be provided for 'all' argument")
        split_dir = os.path.join(DATA_DIR, args.split)
        dyst_names_lst = [
            folder.name for folder in Path(split_dir).iterdir() if folder.is_dir()
        ]
    else:
        dyst_names_lst = args.dysts_names

    print(f"dyst names: {dyst_names_lst}")

    # optionally make plot labels aware of the samples subset (e.g the samples that succeeded or failed ethe tests)
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
        plot_univariate=args.plot_univariate,
        samples_subset_dict=samples_subset_dict,
        plot_name_suffix="failures"
        if args.samples_subset == "failed_samples"
        else None,
    )
