import logging
import tracemalloc
from functools import partial

import hydra
import numpy as np
from dysts.base import DynSys
from memory_profiler import profile
from omegaconf import DictConfig

from scripts.make_skew_systems import (
    DynSysSampler,
    InstabilityEvent,
    OnAttractorInitCondSampler,
    SignedGaussianParamSampler,
    TimeLimitEvent,
    TimeStepEvent,
    additive_coupling_map_factory,
    default_attractor_tests,
    init_skew_system,
    init_trajectory_stats_cache,
)


def setup_sampler(
    cfg: DictConfig, num_ics: int = 1, num_param_perturbations: int = 1
) -> DynSysSampler:
    """Setup the DynSysSampler with minimal test configuration"""
    # Setup events
    event_fns = [
        partial(
            TimeLimitEvent,
            max_duration=cfg.events.max_duration,
            verbose=cfg.events.verbose,
        ),
        partial(
            InstabilityEvent,
            threshold=cfg.events.instability_threshold,
            verbose=cfg.events.verbose,
        ),
        partial(
            TimeStepEvent, min_step=cfg.events.min_step, verbose=cfg.events.verbose
        ),
    ]

    # Setup samplers
    param_sampler = SignedGaussianParamSampler(
        random_seed=cfg.sampling.rseed,
        scale=cfg.sampling.param_scale,
        verbose=cfg.sampling.verbose,
    )

    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=cfg.sampling.reference_traj.length,
        reference_traj_n_periods=cfg.sampling.reference_traj.n_periods,
        reference_traj_transient=cfg.sampling.reference_traj.transient,
        reference_traj_atol=cfg.sampling.reference_traj.atol,
        reference_traj_rtol=cfg.sampling.reference_traj.rtol,
        recompute_standardization=cfg.sampling.standardize,
        random_seed=cfg.sampling.rseed,
        events=event_fns,
        silence_integration_errors=cfg.sampling.silence_integration_errors,
        verbose=int(cfg.sampling.verbose),
    )

    num_periods_lst = np.arange(
        cfg.sampling.num_periods_min, cfg.sampling.num_periods_max + 1
    ).tolist()

    return DynSysSampler(
        rseed=cfg.sampling.rseed,
        num_periods=num_periods_lst,
        num_points=cfg.sampling.num_points,
        num_ics=num_ics,
        num_param_perturbations=num_param_perturbations,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=event_fns,
        verbose=cfg.sampling.verbose,
        split_coords=cfg.sampling.split_coords,
        attractor_tests=default_attractor_tests(),
        validator_transient_frac=cfg.validator.transient_time_frac,
        save_failed_trajs=cfg.validator.save_failed_trajs,
    )


def setup_test_system(
    cfg: DictConfig,
) -> DynSys:
    """
    Setup a single test skew system
    """
    driver, response = "Lorenz", "Rossler"  # Use simple test systems

    # Initialize stats cache
    stats_cache = init_trajectory_stats_cache(
        [driver, response],
        cfg.sampling.num_points,
        cfg.sampling.num_periods_min,
        cfg.sampling.reference_traj.transient,
        atol=cfg.sampling.reference_traj.atol,
        rtol=cfg.sampling.reference_traj.rtol,
    )

    # Setup coupling map
    coupling_map = additive_coupling_map_factory(
        driver, response, stats_cache=stats_cache, **cfg.skew.coupling_map
    )

    return init_skew_system(driver, response, coupling_map_fn=coupling_map)


def profile_with_tracemalloc(
    cfg: DictConfig, sampler: DynSysSampler, system: DynSys
) -> None:
    """Profile memory usage with tracemalloc"""
    tracemalloc.start()

    # Take snapshot before
    snapshot1 = tracemalloc.take_snapshot()

    # Run sampling
    sampler.sample_ensembles(
        systems=[system],
        split="test",
        save_dir=None,  # Don't save during profiling
        standardize=cfg.sampling.standardize,
        use_multiprocessing=False,  # Disable for proper profiling
        silent_errors=cfg.sampling.silence_integration_errors,
        atol=cfg.sampling.atol,
        rtol=cfg.sampling.rtol,
    )

    # Take snapshot after
    snapshot2 = tracemalloc.take_snapshot()

    # Compare and display top differences
    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    logger.info("\n[ Top 10 memory differences ]")
    for stat in top_stats[:10]:
        logger.info(stat)

    tracemalloc.stop()


@profile
def profile_with_memory_profiler(
    cfg: DictConfig, sampler: DynSysSampler, system: DynSys
) -> None:
    """Profile memory usage with memory_profiler"""
    sampler.sample_ensembles(
        systems=[system],
        split="test",
        save_dir=None,
        standardize=cfg.sampling.standardize,
        use_multiprocessing=False,
        silent_errors=cfg.sampling.silence_integration_errors,
        atol=cfg.sampling.atol,
        rtol=cfg.sampling.rtol,
    )


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Initialize components
    logger.info("Setting up sampler and test system...")
    sampler = setup_sampler(cfg, num_ics=1, num_param_perturbations=1)
    system = setup_test_system(cfg)

    # Run profiling
    logger.info("Running tracemalloc profiling...")
    profile_with_tracemalloc(cfg, sampler, system)

    logger.info("\nRunning memory-profiler profiling...")
    profile_with_memory_profiler(cfg, sampler, system)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
