"""
Script to generate and save trajectory ensembles for a given set of dynamical systems.
"""

import argparse
import os

import numpy as np

from dystformer.dyst_data import DystData
from dystformer.sampling import (
    InstabilityEvent,
    OnAttractorInitCondSampler,
    TimeLimitEvent,
)
from dystformer.utils import plot_trajs_multivariate, split_systems


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate and save trajectory ensembles."
    )
    parser.add_argument(
        "--debug-dyst", type=str, help="generate data for a single dynamical system."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.getenv("WORK", ""), "data"),
        help="Directory to save the generated data",
    )
    parser.add_argument(
        "--split-prefix",
        type=str,
        default=None,
        help="Optional prefix for the split names (e.g., 'patchtst' for 'patchtst_train')",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable debug mode for saving failed trajectory ensembles",
    )
    parser.add_argument(
        "--rseed",
        type=int,
        default=999,
        help="Random seed for split, IC, and param samplers",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=60 * 3,
        help="Maximum duration for the TimeLimitEvent",
    )
    parser.add_argument(
        "--instability-threshold",
        type=float,
        default=1e3,
        help="Threshold for the InstabilityEvent",
    )
    parser.add_argument(
        "--param-scale",
        type=float,
        default=0.5,
        help="Scale for the GaussianParamSampler",
    )
    parser.add_argument(
        "--reference-traj-length",
        type=int,
        default=2048,
        help="Reference trajectory length for OnAttractorInitCondSampler",
    )
    parser.add_argument(
        "--reference-traj-transient",
        type=float,
        default=0.2,
        help="Reference trajectory transient for OnAttractorInitCondSampler",
    )
    parser.add_argument(
        "--num-periods",
        type=int,
        default=20,
        help="Number of periods for DystData",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1024 * 4,
        help="Number of points for DystData",
    )
    parser.add_argument(
        "--num-ics",
        type=int,
        default=1,
        help="Number of initial conditions for DystData",
    )
    parser.add_argument(
        "--num-param-perturbations",
        type=int,
        default=10,
        help="Number of parameter perturbations for DystData",
    )
    parser.add_argument(
        "--split-coords",
        action="store_true",
        help="Set to split coordinates into univariate time series",
    )
    parser.add_argument(
        "--no-attractor-tests",
        action="store_false",
        help="Do not apply attractor tests",
    )
    parser.add_argument(
        "--standardize-train",
        action="store_true",
        help="Standardize the train data",
    )
    parser.add_argument(
        "--standardize-test",
        action="store_true",
        help="Standardize the test data",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.3,
        help="Fraction of systems to use for testing",
    )
    parser.add_argument(
        "--sys-class",
        type=str,
        default="continuous",
        choices=["continuous", "continuous_no_delay", "delay", "discrete"],
        help="System class for splitting",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Use multiprocessing for integrating the ensemble",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # generate split of dynamical systems
    test, train = split_systems(
        args.test_split, seed=args.rseed, sys_class=args.sys_class
    )

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=args.max_duration)
    instability_event = InstabilityEvent(threshold=args.instability_threshold)
    events = [time_limit_event, instability_event]

    param_sampler = SignedGaussianParamSampler(
        random_seed=args.rseed,
        scale=args.param_scale,
        verbose=True,
    )
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=args.reference_traj_length,
        reference_traj_transient=args.reference_traj_transient,
        recompute_standardization=True,  # Important!
        events=events,
        verbose=False,
        random_seed=args.rseed,
    )

    dyst_data_generator = DystData(
        rseed=args.rseed,
        num_periods=args.num_periods,
        num_points=args.num_points,
        num_ics=args.num_ics,
        num_param_perturbations=args.num_param_perturbations,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=events,
        verbose=True,
        split_coords=args.split_coords,
        apply_attractor_tests=args.no_attractor_tests,
        attractor_validator_kwargs={
            "verbose": 0,
            "transient_time_frac": 0.05,
            "plot_save_dir": "tests/plots",
        },
        debug_mode=args.debug_mode,
    )

    if args.debug_dyst:
        # Run save_dyst_ensemble on a single system in debug mode
        ensembles = dyst_data_generator._generate_ensembles(
            dysts_names=[args.debug_dyst]
        )
        samples = np.array(
            [ensemble[args.debug_dyst] for ensemble in ensembles if len(ensemble) > 0]
        ).transpose(0, 2, 1)
        plot_trajs_multivariate(
            samples,
            save_dir="figures",
            plot_name=f"{args.debug_dyst}_debug",
        )

    else:
        split_prefix = args.split_prefix + "_" if args.split_prefix else ""

        dyst_data_generator.save_dyst_ensemble(
            dysts_names=train,
            split=f"{split_prefix}train",
            split_failures="failed_attractors_train",
            samples_process_interval=1,
            save_dir=args.data_dir,
            standardize=args.standardize_train,
            use_multiprocessing=args.multiprocessing,
        )
        dyst_data_generator.save_summary(
            os.path.join("outputs", "train_attractor_checks.json"),
        )

        dyst_data_generator.save_dyst_ensemble(
            dysts_names=test,
            split=f"{split_prefix}test",
            split_failures="failed_attractors_test",
            samples_process_interval=1,
            save_dir=args.data_dir,
            standardize=args.standardize_test,
            reset_attractor_validator=True,  # save validator results separately for test
            use_multiprocessing=args.multiprocessing,
        )
        dyst_data_generator.save_summary(
            os.path.join("outputs", "test_attractor_checks.json"),
        )


if __name__ == "__main__":
    main()
