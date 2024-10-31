"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import argparse
import os

from dystformer.sampling import (
    InstabilityEvent,
    OnAttractorInitCondSampler,
    SignedGaussianParamSampler,
    TimeLimitEvent,
    TimeStepEvent,
)
from dystformer.skew_system import SkewData
from dystformer.utils import split_systems


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dysts_names", help="Names of the dynamical systems", nargs="+", type=str
    )
    parser.add_argument(
        "--n_combos", type=int, default=10, help="Number of skew pair combinations"
    )
    parser.add_argument(
        "--couple_phase_space",
        help="Whether to couple phase space",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--couple_flows",
        help="Whether to couple flows",
        type=bool,
        default=False,  # can do both types of coupling
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
        default=60 * 8,
        help="Maximum duration for the TimeLimitEvent",
    )
    parser.add_argument(
        "--instability-threshold",
        type=float,
        default=1e4,
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
        default=4,
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
        "--standardize",
        action="store_true",
        help="Standardize the data",
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
        default="continuous_no_delay",
        choices=["continuous", "continuous_no_delay", "delay", "discrete"],
        help="System class for splitting",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dysts_names = args.dysts_names
    if dysts_names == ["all"]:
        # generate split of dynamical systems
        test_dysts_names, train_dysts_names = split_systems(
            args.test_split, seed=args.rseed, sys_class=args.sys_class
        )
        dysts_names = train_dysts_names + test_dysts_names
    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=args.max_duration)
    instability_event = InstabilityEvent(threshold=args.instability_threshold)
    # NOTE: default min_step=1e-20 may be too small, doesn't catch anything
    time_step_event = TimeStepEvent(min_step=1e-16)
    events = [time_limit_event, instability_event, time_step_event]

    # NOTE: if coupling phase space, need extra caution to make sure g(x+y) is valid rhs i.e. that (x+y) is valid parameter choice for response system
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
        verbose=True,
    )

    print(
        f"Generating {args.n_combos} skew system combinations from {len(dysts_names)} systems"
    )

    split_prefix = args.split_prefix + "_" if args.split_prefix else ""

    skew_data_generator = SkewData(
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
            "transient_time_frac": 0.05,  # don't need long transient time because ic should be on attractor
            "plot_save_dir": "tests/plots",
        },
        debug_mode=args.debug_mode,
        couple_phase_space=args.couple_phase_space,
        couple_flows=args.couple_flows,
    )

    skew_pair_names = skew_data_generator.sample_skew_pairs(dysts_names, args.n_combos)

    print(f"Skew pair names: {skew_pair_names}")

    skew_data_generator.save_dyst_ensemble(
        dysts_names=skew_pair_names,
        split=f"{split_prefix}skew_systems",
        split_failures=f"{split_prefix}failed_skew_systems",
        samples_process_interval=1,
        save_dir=args.data_dir,
        standardize=args.standardize,
    )

    skew_data_generator.save_summary(
        os.path.join("outputs", f"{split_prefix}skew_system_checks.json"),
    )
