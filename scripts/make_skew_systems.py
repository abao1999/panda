"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import json
import logging
import os
from functools import partial
from itertools import permutations
from multiprocessing import Pool
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
from dystformer.coupling_maps import RandomAdditiveCouplingMap
from dystformer.dyst_data import DynSysSampler
from dystformer.events import InstabilityEvent, TimeLimitEvent, TimeStepEvent
from dystformer.sampling import OnAttractorInitCondSampler, SignedGaussianParamSampler
from dystformer.skew_system import SkewProduct
from dystformer.utils import plot_trajs_multivariate


def default_attractor_tests() -> list[Callable]:
    """Builds default attractor tests to check for each trajectory ensemble"""
    tests = [
        partial(check_not_linear, r2_threshold=0.99, eps=1e-10),  # pretty lenient
        partial(check_boundedness, threshold=1e4, max_zscore=15),
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


def plot_single_system(system: DynSys, sys_sampler: DynSysSampler, cfg):
    """Plot a single skew system and its ensembles for debugging"""
    logger.info(f"Generating ensembles for {system.name}")
    ensembles = sys_sampler.sample_ensembles(
        systems=[system],
        save_dir=None,  # NOTE: do not save trajectories in debug mode!
        standardize=cfg.sampling.standardize,
        use_multiprocessing=cfg.sampling.multiprocessing,
        _silent_errors=cfg.sampling.silence_integration_errors,
        atol=cfg.sampling.atol,
        rtol=cfg.sampling.rtol,
    )

    summary_json_path = os.path.join("outputs", "debug_skew_attractor_checks.json")
    logger.info(f"Saving summary for {system.name} to {summary_json_path}")
    sys_sampler.save_summary(summary_json_path)

    with open(summary_json_path, "r") as f:
        summary = json.load(f)

    for subset_name in ["valid_samples", "failed_samples"]:
        samples_subset = summary[subset_name].get(system.name, [])
        if samples_subset == []:
            continue
        coords = np.array(
            [ensembles[i][system.name] for i in samples_subset]
        ).transpose(0, 2, 1)
        coords_response = coords[:, system.driver_dim :]

        plot_trajs_multivariate(
            coords_response,
            samples_subset=samples_subset,
            save_dir="figures",
            plot_name=f"{system.name}_{subset_name}",
            plot_2d_slice=True,
            plot_projections=True,
            standardize=True if not cfg.sampling.standardize else False,
            max_samples=len(coords),
        )


def init_additive_skew_system(
    driver_name: str,
    response_name: str,
    driver_scale: float,
    response_scale: float,
    seed: int | None = None,
    **kwargs,
) -> DynSys:
    """Initialize a skew-product dynamical system with a driver and response system"""
    driver = getattr(flows, driver_name)()
    response = getattr(flows, response_name)()

    coupling_map = RandomAdditiveCouplingMap(
        driver_dim=driver.dimension,
        response_dim=response.dimension,
        driver_scale=driver_scale,
        response_scale=response_scale,
        random_seed=seed,
    )

    return SkewProduct(
        driver=driver, response=response, coupling_map=coupling_map, **kwargs
    )


def sample_skew_systems(
    systems: list[str], num_pairs: int, random_seed: int = 0
) -> list[tuple[str, str]]:
    system_pairs = list(permutations(systems, 2))
    rng = np.random.default_rng(random_seed)
    sampled_pairs = rng.choice(
        len(system_pairs), size=min(num_pairs, len(system_pairs)), replace=False
    )
    return [system_pairs[i] for i in sampled_pairs]


def filter_and_split_skew_systems(
    skew_pairs: list[tuple[str, str]],
    test_split: float = 0.2,
    random_seed: int | None = None,
    scale_cache: dict[str, float] | None = None,
    train_systems: list[str] | None = None,
    test_systems: list[str] | None = None,
    **skew_system_kwargs,
) -> tuple[list[str], list[str]]:
    """Sample skew systems from all pairs of non-skew systems and split into train/test

    TODO: filter skew systems based on non-skew train and test sets, optionally

    Args:
        skew_pairs: List of skew system pairs to sample from
        test_split: Fraction of systems to use for testing
        random_seed: Random seed for reproducibility
        scale_cache: Optional dictionary mapping system names to their RMS flow scales.
            If None, scales are set to 1.0
        train_systems: Optional list of system names to use for training
        test_systems: Optional list of system names to use for testing

    Returns:
        Tuple of (train_systems, test_systems) where each is a list of initialized
        skew product systems
    """
    split_idx = int(len(skew_pairs) * (1 - test_split))
    train_pairs = skew_pairs[:split_idx]
    test_pairs = skew_pairs[split_idx:]

    # if provided, filter out pairs from train and test pairs that contain systems
    # that are not in the train or test sets, then recombine to update valid train/test pairs
    def is_valid_pair(pair: tuple[str, str], filter_list: list[str] | None) -> bool:
        return (
            True
            if filter_list is None
            else all(system in filter_list for system in pair)
        )

    valid_train_pairs = filter(
        lambda pair: is_valid_pair(pair, train_systems), train_pairs
    )
    valid_test_pairs = filter(
        lambda pair: is_valid_pair(pair, test_systems), test_pairs
    )
    invalid_train_pairs = filter(
        lambda pair: not is_valid_pair(pair, train_systems), train_pairs
    )
    invalid_test_pairs = filter(
        lambda pair: not is_valid_pair(pair, test_systems), test_pairs
    )
    train_pairs = list(valid_train_pairs) + list(invalid_test_pairs)
    test_pairs = list(valid_test_pairs) + list(invalid_train_pairs)

    train_systems = [
        init_additive_skew_system(
            driver,
            response,
            1.0 if scale_cache is None else scale_cache[driver],
            1.0 if scale_cache is None else scale_cache[response],
            seed=random_seed,
            **skew_system_kwargs,
        )
        for driver, response in train_pairs
    ]
    test_systems = [
        init_additive_skew_system(
            driver,
            response,
            1.0 if scale_cache is None else scale_cache[driver],
            1.0 if scale_cache is None else scale_cache[response],
            seed=random_seed,
            **skew_system_kwargs,
        )
        for driver, response in test_pairs
    ]

    return train_systems, test_systems


def _compute_system_scale(
    system: str,
    n: int,
    num_periods: int,
    transient: int,
    atol: float,
    rtol: float,
    stiffness: float = 1.0,
) -> tuple[str, float]:
    """Compute RMS scale in flow spacefor a single system"""
    sys = getattr(flows, system)()
    ts, traj = sys.make_trajectory(
        n, pts_per_period=n // num_periods, return_times=True, atol=atol, rtol=rtol
    )
    assert traj is not None, f"{system} should be integrable"
    flow_rms = np.sqrt(
        np.mean(
            [np.max(sys(x, t)) ** 2 for x, t in zip(traj[transient:], ts[transient:])]
        )
    )
    return system, stiffness / flow_rms


def init_trajectory_scale_cache(
    systems: list[str],
    traj_length: int,
    num_periods: int,
    traj_transient: float,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    """Initialize a cache of vector field RMS scales for each system using multiprocessing"""
    _compute_scale_worker = partial(
        _compute_system_scale,
        n=traj_length,
        num_periods=num_periods,
        transient=int(traj_length * traj_transient),
        atol=atol,
        rtol=rtol,
    )
    with Pool() as pool:
        results = pool.map(_compute_scale_worker, systems)
    return dict(results)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    systems = get_attractor_list(sys_class=cfg.sampling.sys_class)

    # events for solve_ivp
    time_limit_event = partial(
        TimeLimitEvent,
        max_duration=cfg.events.max_duration,
        verbose=cfg.events.verbose,
    )
    instability_event = partial(
        InstabilityEvent,
        threshold=cfg.events.instability_threshold,
        verbose=cfg.events.verbose,
    )
    time_step_event = partial(
        TimeStepEvent,
        min_step=cfg.events.min_step,
        verbose=cfg.events.verbose,
    )
    event_fns = [time_limit_event, instability_event, time_step_event]

    # initialize samplers for perturbing systems
    param_sampler = SignedGaussianParamSampler(
        random_seed=cfg.sampling.rseed,
        scale=cfg.sampling.param_scale,
        verbose=cfg.sampling.verbose,
    )
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=cfg.sampling.reference_traj_length,
        reference_traj_n_periods=cfg.sampling.reference_traj_n_periods,
        reference_traj_transient=cfg.sampling.reference_traj_transient,
        reference_traj_atol=cfg.sampling.atol,
        reference_traj_rtol=cfg.sampling.rtol,
        recompute_standardization=cfg.sampling.standardize,  # Important (if standardize=True)
        random_seed=cfg.sampling.rseed,
        events=event_fns,
        silence_integration_errors=cfg.sampling.silence_integration_errors,
        verbose=1,
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
        attractor_tests=default_attractor_tests(),
        attractor_validator_kwargs={
            "verbose": cfg.validator.verbose,
            "transient_time_frac": cfg.validator.transient_time_frac,
            "plot_save_dir": cfg.validator.plot_save_dir,
        },
        save_failed_trajs=cfg.validator.save_failed_trajs,
    )

    ###########################################################################
    # optionally plot a single skew system for debugging, and exit after
    ###########################################################################
    if cfg.sampling.debug_system:
        driver, response = cfg.sampling.debug_system.split("_")
        logger.info(f"Initializing trajectory scale cache for {driver} and {response}")
        scale_cache = init_trajectory_scale_cache(
            [driver, response],
            cfg.sampling.num_points,
            cfg.sampling.num_periods,
            cfg.sampling.reference_traj_transient,
            atol=cfg.sampling.atol,
            rtol=cfg.sampling.rtol,
        )  # type: ignore
        system = init_additive_skew_system(
            driver, response, scale_cache[driver], scale_cache[response]
        )
        plot_single_system(system, sys_sampler, cfg)
        exit()

    # sample skew system train/test splits
    skew_pairs = sample_skew_systems(
        systems, cfg.skew.num_pairs, random_seed=cfg.sampling.rseed
    )

    logger.info(
        f"Sampled {cfg.skew.num_pairs}/{len(systems)*(len(systems)-1)} system pair candidates"
    )

    base_systems = set(sys for pair in skew_pairs for sys in pair)
    logger.info(f"Initializing trajectory scale cache for {len(base_systems)} systems")
    scale_cache = init_trajectory_scale_cache(
        list(base_systems),
        cfg.sampling.num_points,
        cfg.sampling.num_periods,
        cfg.sampling.reference_traj_transient,
        atol=cfg.sampling.atol,
        rtol=cfg.sampling.rtol,
    )

    logger.info(
        f"Attempting to split {len(skew_pairs)} skew pairs into a "
        f"{1-cfg.sampling.test_split:.2f}/{cfg.sampling.test_split:.2f} train/test split"
    )
    train_systems, test_systems = filter_and_split_skew_systems(
        skew_pairs,
        scale_cache=scale_cache,
        test_split=cfg.sampling.test_split,
        random_seed=cfg.sampling.rseed,
    )
    train_prop = len(train_systems) / len(skew_pairs)
    test_prop = len(test_systems) / len(skew_pairs)
    logger.info(
        f"Achieved {len(train_systems)}/{len(test_systems)} = {train_prop:.2f}/{test_prop:.2f}"
        " train/test split after filtering"
    )

    param_dir = (
        os.path.join(cfg.sampling.data_dir, "parameters")
        if cfg.sampling.save_params
        else None
    )

    split_prefix = cfg.sampling.split_prefix + "_" if cfg.sampling.split_prefix else ""
    for split, systems in [("train", train_systems), ("test", test_systems)]:
        sys_sampler.sample_ensembles(
            systems=systems,
            split=f"{split_prefix}{split}",
            split_failures=f"{split_prefix}failed_attractors_{split}",
            samples_process_interval=1,
            save_dir=cfg.sampling.data_dir,
            save_params_dir=f"{param_dir}/{split}" if param_dir else None,
            standardize=cfg.sampling.standardize,
            use_multiprocessing=cfg.sampling.multiprocessing,
            reset_attractor_validator=True,
            silent_errors=cfg.sampling.silence_integration_errors,
            atol=cfg.sampling.atol,
            rtol=cfg.sampling.rtol,
            use_tqdm=False,
        )
        sys_sampler.save_summary(
            os.path.join("outputs", f"{split_prefix}{split}_attractor_checks.json"),
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
