"""
Test util to test individual attractor checks on saved trajectories loaded from Arrow files
"""

import argparse
import os
from functools import partial
from typing import Dict, List, Optional

import numpy as np

from dystformer.attractor import (
    AttractorValidator,
    check_boundedness,
    check_not_fixed_point,
    check_not_trajectory_decay,
)
from dystformer.utils import (
    make_ensemble_from_arrow_dir,
    plot_trajs_multivariate,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def plot_ensemble(
    ensemble: Dict[str, np.ndarray],
    save_dir: str,
    samples_subset: Optional[List[int]] = None,
):
    for dyst_name, dyst_coords_samples in ensemble.items():
        plot_trajs_multivariate(
            dyst_coords_samples,
            save_dir=save_dir,
            plot_name=dyst_name,
            samples_subset=samples_subset,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        help="Names of the dynamical systems",
        nargs="+",
        type=str,
        default=None,
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
    parser.add_argument(
        "--samples_subset",
        help="Indices of samples to use from each trajectory",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.split is None:
        raise ValueError("Split must be provided for loading data")
    dyst_names_lst = args.systems

    if args.samples_subset is not None:
        samples_subset = [int(i) for i in args.samples_subset.split(",")]
        print(f"Using sample subset: {samples_subset}")
    else:
        samples_subset = None

    ### Build attractor validator ###
    validator = AttractorValidator(
        verbose=0, transient_time_frac=0.05, plot_save_dir=None
    )
    validator.add_test_fn(partial(check_boundedness, threshold=1e3, max_num_stds=10))
    validator.add_test_fn(partial(check_not_trajectory_decay, atol=1e-3, tail_prop=0.5))
    validator.add_test_fn(partial(check_not_fixed_point, atol=1e-3, tail_prop=0.1))

    ### Make ensemble from Arrow files ###
    ensemble = make_ensemble_from_arrow_dir(
        DATA_DIR,
        args.split,
        dyst_names_lst=dyst_names_lst,
        one_dim_target=args.one_dim_target,
        samples_subset=samples_subset,
        verbose=True,
    )

    ### Filter ensemble ###
    ensemble, failed_ensemble = validator.filter_ensemble(ensemble)
    print(len(failed_ensemble))

    ### Plot ensemble ###
    plot_ensemble(ensemble, save_dir=args.save_dir, samples_subset=samples_subset)
