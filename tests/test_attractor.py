"""
Test util to test individual attractor checks on saved trajectories loaded from Arrow files
"""

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
from gluonts.dataset.common import FileDataset

from dystformer.attractor import (
    AttractorValidator,
    check_boundedness,
    check_not_fixed_point,
    check_not_trajectory_decay,
)
from dystformer.utils import (
    get_system_filepaths,
    plot_trajs_multivariate,
    stack_and_extract_metadata,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def make_ensemble(
    dyst_names_lst: List[str],
    split: str,
    one_dim_target: bool = False,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    ensemble = {}
    for dyst_name in dyst_names_lst:
        filepaths = get_system_filepaths(dyst_name, DATA_DIR, split)
        if verbose:
            print(f"{dyst_name} filepaths: ", filepaths)
        dyst_coords_samples = []
        for filepath in filepaths:
            # create dataset by reading directly from filepath into FileDataset
            gts_dataset = FileDataset(
                path=Path(filepath),
                freq="h",
                one_dim_target=one_dim_target,
            )
            # extract the coordinates
            dyst_coords, metadata = stack_and_extract_metadata(gts_dataset)
            dyst_coords_samples.append(dyst_coords)
            if verbose:
                print("data shape: ", dyst_coords.shape)
                print("metadata: ", metadata)
                print("IC: ", dyst_coords[:, 0])

        dyst_coords_samples = np.array(dyst_coords_samples)  # type: ignore
        print(dyst_coords_samples.shape)
        ensemble[dyst_name] = dyst_coords_samples
    return ensemble


def plot_ensemble(ensemble: Dict[str, np.ndarray], save_dir: str):
    for dyst_name, dyst_coords_samples in ensemble.items():
        plot_trajs_multivariate(
            dyst_coords_samples, save_dir=save_dir, plot_name=dyst_name
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
        "--save_dir", help="Directory to save plots", type=str, default="tests/figs"
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

    print(f"dyst_names_lst: {dyst_names_lst}")

    ### Build attractor validator ###
    validator = AttractorValidator(
        verbose=0, transient_time_frac=0.05, plot_save_dir=None
    )
    validator.add_test_fn(partial(check_boundedness, threshold=1e3, max_num_stds=10))
    validator.add_test_fn(partial(check_not_fixed_point, atol=1e-3, tail_prop=0.1))
    validator.add_test_fn(check_not_trajectory_decay)
    ### Make ensemble ###
    ensemble = make_ensemble(
        dyst_names_lst, args.split, one_dim_target=args.one_dim_target, verbose=True
    )

    ### Filter ensemble ###
    ensemble, failed_ensemble = validator.filter_ensemble(ensemble)
    print(len(failed_ensemble))

    ### Plot ensemble ###
    plot_ensemble(ensemble, save_dir=args.save_dir)
