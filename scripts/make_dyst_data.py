"""
Script to generate and save trajectory ensembles for a given set of dynamical systems.
"""

import logging
import os
from functools import partial
from typing import Callable, List

import hydra
import numpy as np

from dystformer.attractor import (
    check_boundedness,
    check_lyapunov_exponent,
    check_not_fixed_point,
    check_not_limit_cycle,
    check_not_trajectory_decay,
    check_not_transient,
    check_power_spectrum,
)
from dystformer.dyst_data import DystData
from dystformer.sampling import (
    InstabilityEvent,
    OnAttractorInitCondSampler,
    SignedGaussianParamSampler,
    TimeLimitEvent,
    TimeStepEvent,
)
from dystformer.utils import plot_trajs_multivariate, split_systems


def default_attractor_tests() -> List[Callable]:
    """
    Builds a list of attractor tests to check for each trajectory ensemble.
    """
    print("Setting up callbacks to test attractor properties")
    tests = []
    tests.append(
        partial(check_boundedness, threshold=1e3, max_num_stds=15, save_plot=False)
    )
    tests.append(partial(check_not_fixed_point, atol=1e-3, tail_prop=0.1))
    tests.append(partial(check_not_transient, max_transient_prop=0.2, atol=1e-3))
    tests.append(partial(check_not_trajectory_decay, tail_prop=0.5, atol=1e-3))
    # for STRICT MODE (strict criteria for detecting limit cycles), try:
    # min_prop_recurrences = 0.1, min_counts_per_rtime = 100, min_block_length=50, min_recurrence_time = 10, enforce_endpoint_recurrence = True,
    tests.append(
        partial(
            check_not_limit_cycle,
            tolerance=1e-3,
            min_prop_recurrences=0.1,
            min_counts_per_rtime=100,
            min_block_length=50,
            enforce_endpoint_recurrence=True,
            save_plot=False,
        )
    )
    tests.append(
        partial(
            check_power_spectrum,
            rel_peak_height_threshold=1e-5,
            rel_prominence_threshold=None,
            save_plot=False,
        )
    )
    tests.append(check_lyapunov_exponent)
    return tests


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # generate split of dynamical systems
    test_systems, train_systems = split_systems(
        cfg.dyst_data.test_split,
        seed=cfg.dyst_data.rseed,
        sys_class=cfg.dyst_data.sys_class,
    )
    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=cfg.events.max_duration)
    instability_event = InstabilityEvent(threshold=cfg.events.instability_threshold)
    time_step_event = TimeStepEvent(min_step=cfg.events.min_step)
    events = [time_limit_event, instability_event, time_step_event]

    param_sampler = SignedGaussianParamSampler(
        random_seed=cfg.dyst_data.rseed,
        scale=cfg.dyst_data.param_scale,
        verbose=cfg.dyst_data.verbose,
    )
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=cfg.dyst_data.reference_traj_length,
        reference_traj_transient=cfg.dyst_data.reference_traj_transient,
        recompute_standardization=True,  # Important!
        events=events,
        verbose=cfg.dyst_data.verbose,
        random_seed=cfg.dyst_data.rseed,
    )

    logger.info(f"Dyst data config: {cfg.dyst_data}")
    logger.info(f"Events config: {cfg.events}")
    logger.info(f"Validator config: {cfg.validator}")
    logger.info(
        f"Generating {cfg.dyst_data.num_ics} initial conditions and {cfg.dyst_data.num_param_perturbations} parameter perturbations"
    )
    logger.info(
        f"{len(train_systems)} train systems and {len(test_systems)} test systems with random seed {cfg.dyst_data.rseed}"
    )
    logger.info(f"IC sampler: {ic_sampler}")
    logger.info(f"Param sampler: {param_sampler}")
    logger.info(f"Events: {events}")

    dyst_data_generator = DystData(
        rseed=cfg.dyst_data.rseed,
        num_periods=cfg.dyst_data.num_periods,
        num_points=cfg.dyst_data.num_points,
        num_ics=cfg.dyst_data.num_ics,
        num_param_perturbations=cfg.dyst_data.num_param_perturbations,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=events,
        verbose=cfg.dyst_data.verbose,
        split_coords=cfg.dyst_data.split_coords,
        attractor_validator_kwargs={
            "verbose": cfg.validator.verbose,
            "transient_time_frac": cfg.validator.transient_time_frac,
            "plot_save_dir": cfg.validator.plot_save_dir,
        },
        attractor_tests=default_attractor_tests(),
        save_failed_trajs=cfg.validator.save_failed_trajs,
    )

    if cfg.dyst_data.debug_dyst:
        # Run save_dyst_ensemble on a single system in debug mode
        ensembles = dyst_data_generator._generate_ensembles(
            systems=[cfg.dyst_data.debug_dyst],
            _silent_errors=cfg.dyst_data.silent_errors,
        )

        if any(len(ensemble) == 0 for ensemble in ensembles):
            logger.error(f"No valid trajectories found for {cfg.dyst_data.debug_dyst}")
            return

        samples = np.array(
            [
                ensemble[cfg.dyst_data.debug_dyst]
                for ensemble in ensembles
                if len(ensemble) > 0
            ]
        ).transpose(0, 2, 1)
        plot_trajs_multivariate(
            samples,
            save_dir="figures",
            plot_name=f"{cfg.dyst_data.debug_dyst}_debug",
        )

    else:
        split_prefix = (
            cfg.dyst_data.split_prefix + "_" if cfg.dyst_data.split_prefix else ""
        )

        dyst_data_generator.save_dyst_ensemble(
            systems=train_systems,
            split=f"{split_prefix}train",
            split_failures=f"{split_prefix}failed_attractors_train",
            samples_process_interval=1,
            save_dir=cfg.dyst_data.data_dir,
            standardize=cfg.dyst_data.standardize,
            use_multiprocessing=cfg.dyst_data.multiprocessing,
            _silent_errors=cfg.dyst_data.silent_errors,
        )
        dyst_data_generator.save_summary(
            os.path.join("outputs", f"{split_prefix}train_attractor_checks.json"),
        )

        dyst_data_generator.save_dyst_ensemble(
            systems=test_systems,
            split=f"{split_prefix}test",
            split_failures=f"{split_prefix}failed_attractors_test",
            samples_process_interval=1,
            save_dir=cfg.dyst_data.data_dir,
            standardize=cfg.dyst_data.standardize,
            reset_attractor_validator=True,  # save validator results separately for test
            use_multiprocessing=cfg.dyst_data.multiprocessing,
            _silent_errors=cfg.dyst_data.silent_errors,
        )
        dyst_data_generator.save_summary(
            os.path.join("outputs", f"{split_prefix}test_attractor_checks.json"),
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
