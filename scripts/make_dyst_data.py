"""
Script to generate and save trajectory ensembles for a given set of dynamical systems.
"""

import os

from dystformer.dyst_data import DystData
from dystformer.sampling import (
    GaussianParamSampler,
    InstabilityEvent,
    OnAttractorInitCondSampler,
    TimeLimitEvent,
)
from dystformer.utils import (
    split_systems,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def main():
    # set random seed
    rseed = 999  # we are using same seed for split and ic and param samplers

    # generate split of dynamical systems
    test, train = split_systems(0.3, seed=rseed, sys_class="continuous")

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=60 * 3)  # 2 min time limit
    instability_event = InstabilityEvent(threshold=1e4)
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
        num_ics=3,
        num_param_perturbations=3,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=events,
        verbose=True,
        split_coords=False,  # false for patchtst
        apply_attractor_tests=True,
    )

    # make the train split
    dyst_data_generator.save_dyst_ensemble(
        dysts_names=train,
        split="train",
        samples_process_interval=1,
        save_dir=DATA_DIR,
    )

    # make the test split
    dyst_data_generator.save_dyst_ensemble(
        dysts_names=test,
        split="test",
        samples_process_interval=1,
        save_dir=DATA_DIR,
    )

    dyst_data_generator.save_summary(
        os.path.join("output", "attractor_checks.json"),
    )


if __name__ == "__main__":
    main()
