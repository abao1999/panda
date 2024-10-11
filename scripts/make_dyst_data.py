"""
Script to generate and save trajectory ensembles for a given set of dynamical systems.
"""

import argparse
import os

import numpy as np

from dystformer.dyst_data import DystData
from dystformer.sampling import (
    GaussianParamSampler,
    InstabilityEvent,
    OnAttractorInitCondSampler,
    TimeLimitEvent,
)
from dystformer.utils import plot_trajs_multivariate, split_systems


def main():
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
        default=False,
        help="Enable debug mode for saving failed trajectory ensembles",
    )
    args = parser.parse_args()

    # set random seed
    rseed = 999  # we are using same seed for split and ic and param samplers

    # generate split of dynamical systems
    test, train = split_systems(0.3, seed=rseed, sys_class="delay")

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=60 * 3)  # 2 min time limit
    instability_event = InstabilityEvent(threshold=1e3)
    events = [time_limit_event, instability_event]

    param_sampler = GaussianParamSampler(random_seed=rseed, scale=0.5, verbose=True)
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=2048,
        reference_traj_transient=200,
        events=events,
        verbose=True,
    )

    dyst_data_generator = DystData(
        rseed=rseed,
        num_periods=10,
        num_points=2048,
        num_ics=2,
        num_param_perturbations=3,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=events,
        verbose=True,
        split_coords=False,  # False for multivariate models
        apply_attractor_tests=True,
        standardize=True,
        debug_mode=args.debug_mode,
    )

    if args.debug_dyst:
        # Run save_dyst_ensemble on a single system in debug mode
        ensembles = dyst_data_generator._generate_ensembles(
            dysts_names=[args.debug_dyst]
        )
        samples = np.array(
            [ensemble[args.debug_dyst] for ensemble in ensembles]
        ).transpose(0, 2, 1)
        print(samples.shape)
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
            samples_process_interval=1,
            save_dir=args.data_dir,
        )
        dyst_data_generator.save_dyst_ensemble(
            dysts_names=test,
            split=f"{split_prefix}test",
            samples_process_interval=1,
            save_dir=args.data_dir,
        )
        dyst_data_generator.save_summary(
            os.path.join("outputs", "attractor_checks.json"),
        )


if __name__ == "__main__":
    main()
