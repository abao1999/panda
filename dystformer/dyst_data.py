import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Callable

import dysts.flows as flows
import numpy as np
from dysts.base import BaseDyn
from dysts.sampling import BaseSampler
from dysts.systems import make_trajectory_ensemble
from tqdm import tqdm

from dystformer.attractor import AttractorValidator
from dystformer.utils import demote_from_numpy, process_trajs, timeit

logger = logging.getLogger(__name__)


@dataclass
class DynSysSampler:
    """
    Class to generate and save trajectory ensembles for a given set of dynamical systems.
    Args:
        rseed: random seed for reproducibility
        num_periods: number of periods to generate for each system
        num_points: number of points to generate for each system
        num_ics: number of initial conditions to generate for each system
        num_param_perturbations: number of parameter perturbations to generate for each system
        split_coords: whether to split the coordinates into two dimensions
        events: events to pass to solve_ivp
        attractor_validator_kwargs: kwargs for the attractor validator
        verbose: whether to print verbose output
        save_failed_trajs: flag to save failed trajectory ensembles for debugging
    """

    rseed: int = 999
    num_periods: int = 5
    num_points: int = 1024

    param_sampler: BaseSampler | None = None
    ic_sampler: BaseSampler | None = None
    num_ics: int = 1
    num_param_perturbations: int = 1

    split_coords: bool = True  # by default save trajectories compatible with Chronos
    events: list | None = None
    attractor_validator_kwargs: dict[str, Any] = field(default_factory=dict)
    attractor_tests: list[Callable] | None = None

    verbose: bool = True
    save_failed_trajs: bool = False

    def __post_init__(self) -> None:
        self.failed_integrations = defaultdict(list)
        if self.param_sampler is None:
            assert (
                self.num_param_perturbations == 1
            ), "No parameter sampler provided, but num_param_perturbations > 1"
        if self.ic_sampler is None:
            assert (
                self.num_ics == 1
            ), "No initial condition sampler provided, but num_ics > 1"
        if self.attractor_tests is None and self.num_param_perturbations > 1:
            logger.warning(
                "No attractor tests specified. Parameter perturbations may not result in valid attractors!"
            )
        elif self.attractor_tests is not None:
            self.attractor_validator = AttractorValidator(
                **self.attractor_validator_kwargs,
                tests=self.attractor_tests,
                logger=logger,
            )

    def _prepare_save_directories(
        self,
        save_dir: str | None,
        split: str,
        split_failures: str = "failed_attractors",
    ) -> tuple[str | None, str | None]:
        if save_dir is not None:
            save_dyst_dir = os.path.join(save_dir, split)
            os.makedirs(save_dyst_dir, exist_ok=True)
            logger.info(f"valid attractors will be saved to {save_dyst_dir}")
            if self.save_failed_trajs:
                failed_dyst_dir = os.path.join(save_dir, split_failures)
                os.makedirs(failed_dyst_dir, exist_ok=True)
                logger.info(f"failed attractors will be saved to {failed_dyst_dir}")
            else:
                failed_dyst_dir = None
        else:
            logger.warning("save_dir is None, will not save trajectories.")
            save_dyst_dir = failed_dyst_dir = None
        return save_dyst_dir, failed_dyst_dir

    @timeit(logger=logger)
    def sample_ensembles(
        self,
        systems: list[str] | list[BaseDyn],
        split: str = "train",
        split_failures: str = "failed_attractors",
        samples_process_interval: int = 1,
        save_dir: str | None = None,
        save_params_dir: str | None = None,
        standardize: bool = False,
        use_multiprocessing: bool = True,
        reset_attractor_validator: bool = False,
        **kwargs,
    ) -> list[dict[str, np.ndarray]]:
        """
        Sample perturbed ensembles for a given set of dynamical systems. Optionally,
        save the ensembles to disk and save the parameters to a json file.
        """
        sys_names = [sys if isinstance(sys, str) else sys.name for sys in systems]
        assert len(set(sys_names)) == len(
            sys_names
        ), "Cannot have duplicate system names"
        logger.info(
            f"Making {split} split with {len(systems)} dynamical systems: \n {sys_names}"
        )

        if self.attractor_validator is not None and reset_attractor_validator:
            self.attractor_validator.reset()
            self.failed_integrations.clear()

        save_dyst_dir, failed_dyst_dir = self._prepare_save_directories(
            save_dir, split, split_failures=split_failures
        )
        num_total_samples = self.num_param_perturbations * self.num_ics

        callbacks = [
            self._reset_events_callback,
            self._validate_and_save_ensemble_callback(
                num_total_samples,
                samples_process_interval,
                save_dyst_dir,
                failed_dyst_dir,
                save_params_dir,
            ),
            self.save_failed_integrations_callback,
        ]

        # treat the default params as the zeroth sample
        default_ensemble = make_trajectory_ensemble(
            self.num_points,
            subset=systems,
            pts_per_period=self.num_points // self.num_periods,
            events=self.events,
            use_multiprocessing=use_multiprocessing,
            **kwargs,
        )
        for callback in callbacks[:-1]:  # ignore failed integrations
            callback(0, default_ensemble)

        ensembles = self._generate_ensembles(
            systems,
            postprocessing_callbacks=callbacks,
            standardize=standardize,
            use_multiprocessing=use_multiprocessing,
            **kwargs,
        )
        return ensembles

    def _transform_params_and_ics(
        self,
        system,
        ic_transform=None,
        param_transform=None,
        ic_rng=None,
        param_rng=None,
    ) -> BaseDyn | None:
        """
        Transform the parameters and initial conditions of a system.

        NOTE: If
         - an IC transform or parameter transform is not successful
         - the system is parameterless (len(sys.param_list) == 0)
        the system is not returned (ignored downstream)
        """
        sys = getattr(flows, system)()

        if len(sys.param_list) == 0:
            return None

        success = True
        if param_transform:
            if param_rng is not None:
                param_transform.set_rng(param_rng)
            param_success = sys.transform_params(param_transform)
            success &= param_success
        if ic_transform:
            if ic_rng is not None:
                ic_transform.set_rng(ic_rng)
            ic_success = sys.transform_ic(ic_transform)
            success &= ic_success

        return sys if success else None

    def _init_perturbations(
        self,
        systems: list[str],
        ic_rng: np.random.Generator | None = None,
        param_rng: np.random.Generator | None = None,
    ) -> list[BaseDyn]:
        """
        Pre-initialize the perturbed dyst objects for generation
        """
        ic_rng_stream = [None] * len(systems)
        param_rng_stream = [None] * len(systems)
        if ic_rng is not None:
            ic_rng_stream = ic_rng.spawn(len(systems))
        if param_rng is not None:
            param_rng_stream = param_rng.spawn(len(systems))

        with Pool() as pool:
            transformed_systems = pool.starmap(
                self._transform_params_and_ics,
                [
                    (
                        system,
                        self.ic_sampler,
                        self.param_sampler,
                        ic_rng,
                        param_rng,
                    )
                    for system, ic_rng, param_rng in zip(
                        systems, ic_rng_stream, param_rng_stream
                    )
                ],
            )

        return transformed_systems

    def _generate_ensembles(
        self,
        systems: list[str] | list[BaseDyn],
        postprocessing_callbacks: list[Callable] | None = None,
        **kwargs,
    ) -> list[dict[str, np.ndarray]]:
        """
        Generate trajectory ensembles for a given set of dynamical systems.
        """
        ensembles = []
        pp_rng_stream = np.random.default_rng(self.rseed).spawn(
            self.num_param_perturbations
        )

        total_iterations = self.num_param_perturbations * self.num_ics
        pbar = tqdm(total=total_iterations, desc="Generating ensembles")

        for i, param_rng in enumerate(pp_rng_stream):
            if self.param_sampler is not None:
                self.param_sampler.set_rng(param_rng)

            if self.ic_sampler is not None and hasattr(self.ic_sampler, "clear_cache"):
                self.ic_sampler.clear_cache()  # type: ignore

            ic_rng_stream = param_rng.spawn(self.num_ics)
            for j, ic_rng in enumerate(ic_rng_stream):
                sample_idx = i * len(ic_rng_stream) + j + 1

                pbar.update(1)
                pbar.set_postfix({"param_idx": i, "ic_idx": j})

                # perturb and initialize the system ensemble
                unfiltered_systems = self._init_perturbations(systems, ic_rng=ic_rng)
                excluded_systems = [
                    systems[i]
                    for i in range(len(systems))
                    if unfiltered_systems[i] is None
                ]
                perturbed_systems = [
                    sys for sys in unfiltered_systems if sys is not None
                ]
                assert (
                    len(perturbed_systems) + len(excluded_systems)
                    == len(systems)
                    == len(unfiltered_systems)
                )

                ensemble = make_trajectory_ensemble(
                    self.num_points,
                    subset=perturbed_systems,
                    use_tqdm=True,
                    pts_per_period=self.num_points // self.num_periods,
                    events=self.events,
                    **kwargs,
                )

                # filter out failed integrations
                excluded_systems.extend(
                    key
                    for key, value in ensemble.items()
                    if value is None or np.isnan(value).any()
                )
                ensemble = {
                    key: value
                    for key, value in ensemble.items()
                    if key not in excluded_systems
                }
                ensembles.append(ensemble)

                for callback in postprocessing_callbacks or []:
                    callback(
                        sample_idx,
                        ensemble,
                        excluded_keys=excluded_systems,
                        perturbed_systems=perturbed_systems,
                    )

        return ensembles

    def _reset_events_callback(self, *args, **kwargs) -> None:
        for event in self.events or []:
            if hasattr(event, "reset") and callable(event.reset):
                event.reset()

    def save_failed_integrations_callback(self, sample_idx, ensemble, **kwargs):
        excluded_keys = kwargs.get("excluded_keys", [])
        if len(excluded_keys) > 0:
            logger.warning(f"INTEGRATION FAILED FOR: {excluded_keys}")
            for dyst_name in excluded_keys:
                self.failed_integrations[dyst_name].append(sample_idx)

    def _validate_and_save_ensemble_callback(
        self,
        num_total_samples: int,
        samples_process_interval: int,
        save_dyst_dir: str | None = None,
        failed_dyst_dir: str | None = None,
        save_params_dir: str | None = None,
    ):
        """
        Callback to process and save ensembles and parameters
        """
        ensemble_list = []

        def _callback(sample_idx, ensemble, **kwargs):
            perturbed_systems = kwargs.get("perturbed_systems")
            if len(ensemble.keys()) == 0:
                logger.warning(
                    "No successful trajectories for this sample. Skipping, will not save to arrow files."
                )
                return

            ensemble_list.append(ensemble)

            is_last_sample = (sample_idx + 1) == num_total_samples
            if ((sample_idx + 1) % samples_process_interval) == 0 or is_last_sample:
                self._process_and_save_ensemble(
                    ensemble_list,
                    sample_idx,
                    perturbed_systems=perturbed_systems,
                    save_dyst_dir=save_dyst_dir,
                    failed_dyst_dir=failed_dyst_dir,
                    save_params_dir=save_params_dir,
                )
                ensemble_list.clear()

        return _callback

    def _process_and_save_ensemble(
        self,
        ensemble_list: list[dict[str, np.ndarray]],
        sample_idx: int,
        perturbed_systems: list[BaseDyn] | None = None,
        save_dyst_dir: str | None = None,
        failed_dyst_dir: str | None = None,
        save_params_dir: str | None = None,
    ) -> None:
        """
        Process the ensemble list by checking for valid attractors and filtering out invalid ones.
        Also, transposes and stacks trajectories to get shape (num_samples, num_dims, num_timesteps).
        """
        # transpose and stack to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
        ensemble = {
            key: np.stack(
                [d[key] for d in ensemble_list if key in d], axis=0
            ).transpose(0, 2, 1)
            for key in [key for d in ensemble_list for key in d.keys()]
        }

        if self.attractor_validator is not None:
            ensemble, failed_ensemble = (
                self.attractor_validator.multiprocessed_filter_ensemble(
                    ensemble, first_sample_idx=sample_idx
                )
            )
        else:
            failed_ensemble = {}

        if save_dyst_dir is not None:
            process_trajs(
                save_dyst_dir,
                ensemble,
                split_coords=self.split_coords,
                verbose=self.verbose,
            )

        if failed_dyst_dir is not None:
            process_trajs(
                failed_dyst_dir,
                failed_ensemble,
                split_coords=self.split_coords,
                verbose=self.verbose,
            )

        if save_params_dir is not None and perturbed_systems is not None:
            successful_systems = [
                sys for sys in perturbed_systems if sys.name in ensemble.keys()
            ]
            failed_systems = [
                sys for sys in perturbed_systems if sys.name in failed_ensemble.keys()
            ]
            self._save_parameters(
                successful_systems, os.path.join(save_params_dir, "successes.json")
            )
            self._save_parameters(
                failed_systems, os.path.join(save_params_dir, "failures.json")
            )

    def _save_parameters(
        self, perturbed_systems: list[BaseDyn], save_path: str | None = None
    ) -> None:
        if save_path is None or len(perturbed_systems) == 0:
            return

        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                param_dict = json.load(f)
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            param_dict = {}

        for sys in perturbed_systems:
            if sys.name not in param_dict:
                param_dict[sys.name] = []

            if "_" in sys.name:  # for skew systems
                serialized_params = [
                    list(map(demote_from_numpy, sys.driver.param_list)),
                    list(map(demote_from_numpy, sys.response.param_list)),
                ]
            else:
                serialized_params = list(map(demote_from_numpy, sys.param_list))

            param_dict[sys.name].append(serialized_params)

        with open(save_path, "w") as f:
            json.dump(param_dict, f, indent=4)

    def save_summary(self, save_json_path: str):
        """
        Save a summary of valid attractor counts and failed checks to a json file.
        """
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        logger.info(f"Saving summary to {save_json_path}")

        if self.attractor_validator is None:
            summary_dict = {"failed_integrations": self.failed_integrations}

        else:
            valid_dyst_counts = self.attractor_validator.valid_dyst_counts
            failed_checks = self.attractor_validator.failed_checks
            failed_samples = self.attractor_validator.failed_samples
            valid_samples = self.attractor_validator.valid_samples
            summary_dict = {
                "num_parameter_successes": sum(
                    len(np.unique(np.array(sample_inds).astype(int) // self.num_ics))
                    for sample_inds in valid_samples.values()
                ),
                "num_total_candidates": self.num_param_perturbations
                * len(
                    valid_samples.keys()
                    | failed_samples.keys()
                    | self.failed_integrations.keys()
                ),
                "valid_dyst_counts": valid_dyst_counts,
                "failed_checks": failed_checks,
                "failed_integrations": self.failed_integrations,
                "failed_samples": failed_samples,
                "valid_samples": valid_samples,
            }

        with open(save_json_path, "w") as f:
            json.dump(summary_dict, f, indent=4)
