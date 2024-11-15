"""
Script to generate and save trajectory ensembles for a given set of dynamical systems.
"""

import logging
import os

import hydra
import numpy as np

from dystformer.dyst_data import DystData
from dystformer.sampling import (
    InstabilityEvent,
    OnAttractorInitCondSampler,
    SignedGaussianParamSampler,
    TimeLimitEvent,
    TimeStepEvent,
)
from dystformer.utils import plot_trajs_multivariate, split_systems


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
        apply_attractor_tests=cfg.validator.enable,
        attractor_validator_kwargs={
            "verbose": cfg.validator.verbose,
            "transient_time_frac": cfg.validator.transient_time_frac,
            "plot_save_dir": cfg.validator.plot_save_dir,
        },
        save_failed_trajs=cfg.validator.save_failed_trajs,
    )

    if cfg.dyst_data.debug_dyst:
        # Run save_dyst_ensemble on a single system in debug mode
        ensembles = dyst_data_generator._generate_ensembles(
            systems=[cfg.dyst_data.debug_dyst]
        )
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
        )
        dyst_data_generator.save_summary(
            os.path.join("outputs", f"{split_prefix}test_attractor_checks.json"),
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
