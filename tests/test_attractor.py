"""
Test the parameter perturbations and attractor validation, and save a summary to a json file.
"""

import argparse
import os

from dystformer.dyst_data import DystData
from dystformer.sampling import (
    GaussianParamSampler,
    InstabilityEvent,
    OnAttractorInitCondSampler,
    TimeLimitEvent,
)
from dystformer.utils import split_systems

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


if __name__ == "__main__":
    # For testing select systems
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dysts_names", help="Names of the dynamical systems", nargs="+", type=str
    )
    args = parser.parse_args()
    dysts_names = args.dysts_names

    if dysts_names == ["all"]:
        _, dysts_names = split_systems(0.3, seed=999, sys_class="continuous")

    print("dysts_names: ", dysts_names)

    # set random seed
    rseed = 999  # we are using same seed for split and ic and param samplers

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=60 * 4)  # 4 min time limit
    instability_event = InstabilityEvent(threshold=1e4)
    events = [time_limit_event, instability_event]

    param_sampler = GaussianParamSampler(random_seed=rseed, scale=0.5, verbose=True)
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=1024,
        reference_traj_transient=200,
        events=events,
        verbose=True,
    )

    dyst_data_generator = DystData(
        rseed=rseed,
        num_periods=20,
        num_points=1024 * 4,  # Note: 3x the default
        num_ics=1,
        num_param_perturbations=4,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=events,
        verbose=True,
        split_coords=False,  # false for patchtst
        apply_attractor_tests=True,
        attractor_validator_kwargs={
            "verbose": 1,
            "plot_save_dir": None,
        },
        debug_mode=True,  # save the failed attractors
    )

    dyst_data_generator.save_dyst_ensemble(
        dysts_names=dysts_names,
        split="debug",
        samples_process_interval=1,
        save_dir=DATA_DIR,
    )

    dyst_data_generator.save_summary(
        os.path.join("tests", "attractor_checks.json"),
    )
