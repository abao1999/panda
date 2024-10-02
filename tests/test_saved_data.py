"""
Load saved Arrow data files and plot the trajectories.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from gluonts.dataset.common import FileDataset

from dystformer.utils import (
    get_dyst_filepaths,
    plot_trajs_multivariate,
    stack_and_extract_metadata,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dyst_name", help="Name of the dynamical system", type=str)
    parser.add_argument("--split", help="Split of the data", type=str, default=None)
    parser.add_argument(
        "--ics_per_param", help="Num ics per param perturbation", type=int, default=1
    )
    args = parser.parse_args()

    if args.dyst_name == "all":
        # get all folder names in DATA_DIR/{split}
        if args.split is None:
            raise ValueError("Split must be provided for 'all' argument")
        split_dir = os.path.join(DATA_DIR, args.split)
        dyst_names_lst = [
            folder.name for folder in Path(split_dir).iterdir() if folder.is_dir()
        ]
    else:
        dyst_names_lst = [args.dyst_name]

    print(f"dyst names: {dyst_names_lst}")

    for dyst_name in dyst_names_lst:
        filepaths = get_dyst_filepaths(dyst_name, split=args.split)
        print(f"{dyst_name} filepaths: ", filepaths)

        # NOTE: this is same as accumulate_dyst_samples in tests/test_augmentations.py
        dyst_coords_samples = []
        for filepath in filepaths:
            # create dataset by reading directly from filepath into FileDataset
            gts_dataset = FileDataset(
                path=Path(filepath),
                freq="h",
                one_dim_target=False,  # NOTE: one_dim_target is important!
            )  # TODO: consider other frequencies?

            # extract the coordinates
            dyst_coords, metadata = stack_and_extract_metadata(
                gts_dataset,
                one_dim_target=False,  # NOTE: one_dim_target mportant!
            )

            dyst_coords_samples.append(dyst_coords)

            print("data shape: ", dyst_coords.shape)
            print("metadata: ", metadata)
            print("IC: ", dyst_coords[:, 0])

        dyst_coords_samples = np.array(dyst_coords_samples)  # type: ignore
        print(dyst_coords_samples.shape)

        # plot the trajectories
        plot_trajs_multivariate(
            dyst_coords_samples,
            save_dir="tests/figs",
            plot_name=dyst_name,
            sample_param_interval=args.ics_per_param,
        )
