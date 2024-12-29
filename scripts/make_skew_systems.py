"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import logging
import os
from functools import partial
from itertools import permutations
from typing import Callable

import dysts.flows as flows
import hydra
import numpy as np
from dysts.systems import DynSys, get_attractor_list

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
from dystformer.skew_system import SkewProduct
from dystformer.utils import plot_trajs_multivariate


def init_skew_system(drive_name: str, response_name: str) -> DynSys:
    driver = getattr(flows, drive_name)()
    response = getattr(flows, response_name)()
    return SkewProduct(driver=driver, response=response)


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


def sample_skew_systems(systems: list[str], cfg) -> tuple[list[DynSys], list[DynSys]]:
    system_pairs = list(permutations(systems, 2))

    # Randomly sample 3 system pairs
    n_combos = 3
    rng = np.random.default_rng(cfg.sampling.rseed)
    system_pairs = rng.choice(system_pairs, size=n_combos, replace=False)
    logger.info(f"Generated {len(system_pairs)} system pairs")

    np.random.seed(cfg.sampling.rseed)
    np.random.shuffle(system_pairs)
    split_idx = int(len(system_pairs) * (1 - cfg.sampling.test_split))
    train_pairs = system_pairs[:split_idx]
    test_pairs = system_pairs[split_idx:]

    train_systems = [
        init_skew_system(driver, response) for driver, response in train_pairs
    ]
    test_systems = [
        init_skew_system(driver, response) for driver, response in test_pairs
    ]

    return train_systems, test_systems


def plot_single_system(system: DynSys, sys_sampler: DynSysSampler, cfg):
    default_traj = system.make_trajectory(
        cfg.sampling.num_points,
        pts_per_period=cfg.sampling.num_points // cfg.sampling.num_periods,
    )
    ensembles = sys_sampler._generate_ensembles(
        systems=[system],
        use_multiprocessing=cfg.sampling.multiprocessing,
        _silent_errors=cfg.sampling.silence_integration_errors,
    )

    samples = np.array(
        [default_traj]
        + [ensemble[system.name] for ensemble in ensembles if len(ensemble) > 0]
    ).transpose(0, 2, 1)
    plot_trajs_multivariate(
        samples,
        save_dir="figures",
        plot_name=f"{system.name}_debug",
        plot_2d_slice=True,
        plot_projections=True,
        max_samples=len(ensembles),
    )


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    systems = get_attractor_list(sys_class="continuous_no_delay")

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=cfg.events.max_duration)
    instability_event = InstabilityEvent(threshold=cfg.events.instability_threshold)
    time_step_event = TimeStepEvent(min_step=cfg.events.min_step)
    events = [time_limit_event, instability_event, time_step_event]

    param_sampler = SignedGaussianParamSampler(
        random_seed=cfg.sampling.rseed,
        scale=cfg.sampling.param_scale,
        verbose=cfg.sampling.verbose,
    )
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=cfg.sampling.reference_traj_length,
        reference_traj_transient=cfg.sampling.reference_traj_transient,
        recompute_standardization=cfg.sampling.standardize,  # Important (if standardize=True)
        events=events,
        silence_integration_errors=cfg.sampling.silence_integration_errors,
        verbose=1,
    )

    split_prefix = cfg.sampling.split_prefix + "_" if cfg.sampling.split_prefix else ""

    sys_sampler = DynSysSampler(
        rseed=cfg.sampling.rseed,
        num_periods=cfg.sampling.num_periods,
        num_points=cfg.sampling.num_points,
        num_ics=cfg.sampling.num_ics,
        num_param_perturbations=cfg.sampling.num_param_perturbations,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=events,
        verbose=cfg.sampling.verbose,
        split_coords=cfg.sampling.split_coords,
        attractor_tests=default_attractor_tests(),
        attractor_validator_kwargs={
            "verbose": cfg.validator.verbose,
            "transient_time_frac": cfg.validator.transient_time_frac,
            "plot_save_dir": cfg.validator.plot_save_dir,
        },
        save_failed_trajs=cfg.validator.save_failed_trajs,
    )

    if cfg.sampling.debug_system:
        system = init_skew_system(*cfg.sampling.debug_system.split("_"))
        plot_single_system(system, sys_sampler, cfg)
    else:
        train_systems, test_systems = sample_skew_systems(systems, cfg)

        split_prefix = (
            cfg.sampling.split_prefix + "_" if cfg.sampling.split_prefix else ""
        )

        sys_sampler.sample_ensembles(
            systems=train_systems,
            split=f"{split_prefix}train",
            split_failures=f"{split_prefix}failed_attractors_train",
            samples_process_interval=1,
            save_dir=cfg.sampling.data_dir,
            standardize=cfg.sampling.standardize,
            use_multiprocessing=cfg.sampling.multiprocessing,
        )
        sys_sampler.save_summary(
            os.path.join("outputs", f"{split_prefix}train_attractor_checks.json"),
        )

        sys_sampler.sample_ensembles(
            systems=test_systems,
            split=f"{split_prefix}test",
            split_failures=f"{split_prefix}failed_attractors_test",
            samples_process_interval=1,
            save_dir=cfg.sampling.data_dir,
            standardize=cfg.sampling.standardize,
            reset_attractor_validator=True,  # save validator results separately for test
            use_multiprocessing=cfg.sampling.multiprocessing,
        )
        sys_sampler.save_summary(
            os.path.join("outputs", f"{split_prefix}test_attractor_checks.json"),
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
