"""
Script to generate and save trajectory ensembles for a given set of dynamical systems.
"""

import logging
import os
from functools import partial
from typing import Callable

import dysts.flows as flows
import hydra
import numpy as np

from dystformer.attractor import (
    check_boundedness,
    check_lyapunov_exponent,
    check_not_fixed_point,
    check_not_limit_cycle,
    check_not_linear,
    check_power_spectrum,
)
from dystformer.dyst_data import DynSysSampler
from dystformer.sampling import (
    InstabilityEvent,
    OnAttractorInitCondSampler,
    SignedGaussianParamSampler,
    TimeLimitEvent,
    TimeStepEvent,
)
from dystformer.utils import plot_trajs_multivariate, split_systems


def default_attractor_tests() -> list[Callable]:
    """
    Builds a list of attractor tests to check for each trajectory ensemble.
    """
    tests = [
        partial(check_not_linear, r2_threshold=0.99, eps=1e-10),  # pretty lenient
        partial(check_boundedness, threshold=1e3, max_zscore=15),
        partial(check_not_fixed_point, atol=1e-3, tail_prop=0.1),
        # for STRICT MODE (strict criteria for detecting limit cycles), try:
        # min_prop_recurrences = 0.1, min_counts_per_rtime = 100, min_block_length=50, min_recurrence_time = 10, enforce_endpoint_recurrence = True,
        partial(
            check_not_limit_cycle,
            tolerance=1e-3,
            min_prop_recurrences=0.1,
            min_counts_per_rtime=200,
            min_block_length=50,
            enforce_endpoint_recurrence=True,
        ),
        partial(
            check_power_spectrum, rel_peak_height=1e-5, rel_prominence=1e-5, min_peaks=3
        ),
        partial(check_lyapunov_exponent, traj_len=150),
    ]
    return tests


def plot_single_system(sys_name: str, sys_sampler: DynSysSampler, cfg):
    default_traj = getattr(flows, sys_name)().make_trajectory(
        cfg.sampling.num_points,
        pts_per_period=cfg.sampling.num_points // cfg.sampling.num_periods,
    )
    ensembles = sys_sampler._generate_ensembles(
        systems=[sys_name],
        use_multiprocessing=cfg.sampling.multiprocessing,
        _silent_errors=cfg.sampling.silence_integration_errors,
    )

    samples = np.array(
        [default_traj]
        + [ensemble[sys_name] for ensemble in ensembles if len(ensemble) > 0]
    ).transpose(0, 2, 1)
    plot_trajs_multivariate(
        samples,
        save_dir="figures",
        plot_name=f"{sys_name}_debug",
        plot_2d_slice=True,
        plot_projections=True,
    )


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_systems, train_systems = split_systems(
        cfg.sampling.test_split,
        seed=cfg.sampling.rseed,
        sys_class=cfg.sampling.sys_class,
    )

    time_limit_event = TimeLimitEvent(
        max_duration=cfg.events.max_duration, verbose=cfg.events.verbose
    )
    instability_event = partial(
        InstabilityEvent,
        threshold=cfg.events.instability_threshold,
        verbose=cfg.events.verbose,
    )
    time_step_event = TimeStepEvent(
        min_step=cfg.events.min_step, verbose=cfg.events.verbose
    )
    event_fns = [time_limit_event, time_step_event, instability_event]

    param_sampler = SignedGaussianParamSampler(
        random_seed=cfg.sampling.rseed,
        scale=cfg.sampling.param_scale,
        sign_match_probability=cfg.sampling.sign_match_probability,
        ignore_probability=cfg.sampling.ignore_probability,
        verbose=cfg.sampling.verbose,
    )
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=cfg.sampling.reference_traj_length,
        reference_traj_transient=cfg.sampling.reference_traj_transient,
        recompute_standardization=True,  # Important!
        events=event_fns,
        verbose=cfg.sampling.verbose,
        random_seed=cfg.sampling.rseed,
        silence_integration_errors=cfg.sampling.silence_integration_errors,
    )

    sys_sampler = DynSysSampler(
        rseed=cfg.sampling.rseed,
        num_periods=cfg.sampling.num_periods,
        num_points=cfg.sampling.num_points,
        num_ics=cfg.sampling.num_ics,
        num_param_perturbations=cfg.sampling.num_param_perturbations,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=event_fns,
        verbose=cfg.sampling.verbose,
        split_coords=cfg.sampling.split_coords,
        attractor_validator_kwargs={
            "verbose": cfg.validator.verbose,
            "transient_time_frac": cfg.validator.transient_time_frac,
            "plot_save_dir": cfg.validator.plot_save_dir,
        },
        attractor_tests=default_attractor_tests(),
        save_failed_trajs=cfg.validator.save_failed_trajs,
    )

    ###########################################################################
    # Run save_dyst_ensemble on a single system in debug mode
    ###########################################################################
    if cfg.sampling.debug_system:
        plot_single_system(cfg.sampling.debug_system, sys_sampler, cfg)
        exit()

    split_prefix = cfg.sampling.split_prefix + "_" if cfg.sampling.split_prefix else ""

    if cfg.sampling.save_params:
        param_dir = os.path.join(cfg.sampling.data_dir, "parameters")
    else:
        param_dir = None

    _ = sys_sampler.sample_ensembles(
        systems=train_systems,
        split=f"{split_prefix}train",
        split_failures=f"{split_prefix}failed_attractors_train",
        samples_process_interval=1,
        save_dir=cfg.sampling.data_dir,
        save_params_dir=f"{param_dir}/train" if param_dir else None,
        standardize=cfg.sampling.standardize,
        use_multiprocessing=cfg.sampling.multiprocessing,
        _silent_errors=cfg.sampling.silence_integration_errors,
    )
    sys_sampler.save_summary(
        os.path.join("outputs", f"{split_prefix}train_attractor_checks.json"),
    )

    _ = sys_sampler.sample_ensembles(
        systems=test_systems,
        split=f"{split_prefix}test",
        split_failures=f"{split_prefix}failed_attractors_test",
        samples_process_interval=1,
        save_dir=cfg.sampling.data_dir,
        save_params_dir=f"{param_dir}/test" if param_dir else None,
        standardize=cfg.sampling.standardize,
        reset_attractor_validator=True,  # save validator results separately for test
        use_multiprocessing=cfg.sampling.multiprocessing,
        _silent_errors=cfg.sampling.silence_integration_errors,
    )
    sys_sampler.save_summary(
        os.path.join("outputs", f"{split_prefix}test_attractor_checks.json"),
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
