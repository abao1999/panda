"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import logging
import os

import hydra

from dystformer.sampling import (
    InstabilityEvent,
    OnAttractorInitCondSampler,
    SignedGaussianParamSampler,
    TimeLimitEvent,
    TimeStepEvent,
)
from dystformer.skew_system import SkewData
from dystformer.utils import split_systems


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # generate split of dynamical systems
    test_systems, train_systems = split_systems(
        cfg.dyst_data.test_split,
        seed=cfg.dyst_data.rseed,
        sys_class=cfg.dyst_data.sys_class,
    )
    if cfg.skew.use_all_systems:
        system_names = train_systems + test_systems
    else:
        system_names = train_systems

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
        recompute_standardization=cfg.dyst_data.standardize,  # Important (if standardize=True)
        events=events,
        verbose=cfg.dyst_data.verbose,
    )

    logger.info(f"Dyst data config: {cfg.dyst_data}")
    logger.info(f"Events config: {cfg.events}")
    logger.info(f"Validator config: {cfg.validator}")
    logger.info(f"Skew config: {cfg.skew}")
    logger.info(
        f"Generating {cfg.skew.n_combos} skew system combinations from {len(system_names)} systems with random seed {cfg.dyst_data.rseed}"
    )
    logger.info(f"IC sampler: {ic_sampler}")
    logger.info(f"Param sampler: {param_sampler}")

    split_prefix = (
        cfg.dyst_data.split_prefix + "_" if cfg.dyst_data.split_prefix else ""
    )

    skew_data_generator = SkewData(
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
        apply_attractor_tests=cfg.validator.enable,
        attractor_validator_kwargs={
            "verbose": cfg.validator.verbose,
            "transient_time_frac": cfg.validator.transient_time_frac,  # don't need long transient time because ic should be on attractor
            "plot_save_dir": cfg.validator.plot_save_dir,
        },
        save_failed_trajs=cfg.validator.save_failed_trajs,
        couple_phase_space=cfg.skew.couple_phase_space,
        couple_flows=cfg.skew.couple_flows,
    )

    skew_pair_names = skew_data_generator.sample_skew_pairs(
        system_names, cfg.skew.n_combos
    )

    print(f"Skew pair names: {skew_pair_names}")

    skew_data_generator.save_dyst_ensemble(
        dysts_names=skew_pair_names,
        split=f"{split_prefix}skew_systems",
        split_failures=f"{split_prefix}failed_skew_systems",
        samples_process_interval=1,
        save_dir=cfg.dyst_data.data_dir,
        standardize=cfg.dyst_data.standardize,
    )

    skew_data_generator.save_summary(
        os.path.join("outputs", f"{split_prefix}skew_system_checks.json"),
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
