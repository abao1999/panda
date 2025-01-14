import json
import logging
import os
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import starmap
from multiprocessing import Manager, Pool
from typing import Callable

import dysts.flows as flows
import numpy as np
from dysts.base import BaseDyn
from dysts.sampling import BaseSampler
from dysts.systems import make_trajectory_ensemble
from tqdm import tqdm

from dystformer.attractor import AttractorValidator
from dystformer.sampling import OnAttractorInitCondSampler
from dystformer.skew_system import SkewProduct
from dystformer.utils import dict_demote_from_numpy, process_trajs, timeit

logger = logging.getLogger(__name__)


@contextmanager
def managed_cache(
    sampler: OnAttractorInitCondSampler | None, use_multiprocessing: bool
):
    """Context manager to handle shared cache for OnAttractorInitCondSampler."""
    if use_multiprocessing and isinstance(sampler, OnAttractorInitCondSampler):
        with Manager() as manager:
            sampler.trajectory_cache = manager.dict()  # type: ignore
            try:
                yield
            finally:
                sampler.clear_cache()
    else:
        yield


@dataclass
class DynSysSampler:
    """
    Class to generate and save trajectory ensembles for a given set of dynamical systems.
    Args:
        rseed: random seed for reproducibility
        num_periods: number of periods to generate for each system
        num_points: number of time points to generate for each system
        param_sampler: parameter sampler, samples parameters for each system
        ic_sampler: initial condition sampler, samples initial conditions for each system
        num_ics: number of initial conditions to sample for each system
        num_param_perturbations: number of parameter perturbations to sample for each system
        split_coords: whether to split the coordinates by dimension (univariate) or not (multivariate)
        events: list of solve_ivp events to use for numerical integration
        attractor_validator_kwargs: kwargs for the attractor validator
        attractor_tests: list of tests to use for attractor validator
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
    events: list[Callable[[float, np.ndarray], float]] | None = None

    validator_transient_frac: float = 0.05
    attractor_tests: list[Callable] | None = None

    verbose: bool = True
    save_failed_trajs: bool = False

    def __post_init__(self) -> None:
        self.failed_integrations = defaultdict(list)
        self.rng = np.random.default_rng(self.rseed)
        if self.param_sampler is None:
            assert self.num_param_perturbations == 1, (
                "No parameter sampler provided, but num_param_perturbations > 1"
            )
        if self.ic_sampler is None:
            assert self.num_ics == 1, (
                "No initial condition sampler provided, but num_ics > 1"
            )
        if self.attractor_tests is None and self.num_param_perturbations > 1:
            logger.warning(
                "No attractor tests specified. Parameter perturbations may not result in valid attractors!"
            )
        elif self.attractor_tests is not None:
            self.attractor_validator = AttractorValidator(
                transient_time_frac=self.validator_transient_frac,
                tests=self.attractor_tests,
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
        save_traj_stats_dir: str | None = None,
        standardize: bool = False,
        use_multiprocessing: bool = True,
        silent_errors: bool = False,
        reset_attractor_validator: bool = False,
        **kwargs,
    ) -> list[dict[str, np.ndarray]]:
        """
        Sample perturbed ensembles for a given set of dynamical systems. Optionally,
        save the ensembles to disk and save the parameters to a json file.
        """
        sys_names = [sys if isinstance(sys, str) else sys.name for sys in systems]
        assert len(set(sys_names)) == len(sys_names), (
            "Cannot have duplicate system names"
        )
        if save_dir is not None:
            logger.info(
                f"Making {split} split with {len(systems)} dynamical systems"
                f" (showing first {min(5, len(sys_names))}): \n {sys_names[:5]}"
            )
        is_all_basedyn = all(isinstance(sys, BaseDyn) for sys in systems)

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
                save_traj_stats_dir,
            ),
            self.save_failed_integrations_callback,
        ]

        # treat the default params as the zeroth sample
        logger.info("Generating default ensemble...")
        default_ensemble = make_trajectory_ensemble(
            self.num_points,
            subset=systems,
            pts_per_period=self.num_points // self.num_periods,
            event_fns=self.events,
            use_multiprocessing=use_multiprocessing,
            silent_errors=silent_errors,
            **kwargs,
        )
        failed_integrations = [
            key
            for key, value in default_ensemble.items()
            if value is None or np.isnan(value).any()
        ]
        default_ensemble = {
            key: value
            for key, value in default_ensemble.items()
            if key not in failed_integrations
        }
        for callback in callbacks[:-1]:  # ignore failed integrations
            callback(
                0,
                default_ensemble,
                excluded_keys=failed_integrations,
                perturbed_systems=systems if is_all_basedyn else None,
            )

        logger.info("Generating perturbed ensembles...")
        ensembles = self._generate_ensembles(
            systems,
            postprocessing_callbacks=callbacks,
            standardize=standardize,
            use_multiprocessing=use_multiprocessing,
            silent_errors=silent_errors,
            **kwargs,
        )
        ensembles.insert(0, default_ensemble)

        return ensembles

    def _transform_params_and_ics(
        self,
        system: BaseDyn | str,
        ic_transform: Callable | None = None,
        param_transform: Callable | None = None,
        ic_rng: np.random.Generator | None = None,
        param_rng: np.random.Generator | None = None,
    ) -> BaseDyn | None:
        """
        Transform the parameters and initial conditions of a system.

        NOTE: If
         - an IC transform or parameter transform is not successful
         - the system is parameterless (len(sys.param_list) == 0)
        the system is not returned (ignored downstream)
        """
        sys = getattr(flows, system)() if isinstance(system, str) else system

        if hasattr(sys, "param_list") and len(sys.param_list) == 0:
            return None

        success = True
        if param_transform is not None:
            if param_rng is not None:  # unsafe, address later
                param_transform.set_rng(param_rng)
            param_success = sys.transform_params(param_transform)
            success &= param_success
        if ic_transform is not None:
            if ic_rng is not None:  # unsafe, address later
                ic_transform.set_rng(ic_rng)
            ic_success = sys.transform_ic(ic_transform)
            success &= ic_success

        return sys if success else None

    def _init_perturbations(
        self,
        systems: list[str | BaseDyn],
        ic_rng: np.random.Generator | None = None,
        param_rng: np.random.Generator | None = None,
        perturb_params: bool = False,
        perturb_ics: bool = False,
        use_multiprocessing: bool = True,
    ) -> list[BaseDyn]:
        """
        Pre-initialize the perturbed dyst objects for generation
        """
        assert all(sys is not None for sys in systems), "systems cannot contain None"

        ic_rng_stream = [None] * len(systems)
        if ic_rng is not None:
            ic_rng_stream = ic_rng.spawn(len(systems))

        param_rng_stream = [None] * len(systems)
        if param_rng is not None:
            param_rng_stream = param_rng.spawn(len(systems))

        param_transform = self.param_sampler if perturb_params else None
        ic_transform = self.ic_sampler if perturb_ics else None

        args = (
            (system, ic_transform, param_transform, ic_rng, param_rng)
            for system, ic_rng, param_rng in zip(
                systems, ic_rng_stream, param_rng_stream
            )
        )

        with Pool() if use_multiprocessing else nullcontext() as pool:
            map_fn = pool.starmap if use_multiprocessing else starmap  # type: ignore
            transformed_systems = list(map_fn(self._transform_params_and_ics, args))

        return transformed_systems

    def _generate_ensembles(
        self,
        systems: list[str | BaseDyn],
        use_multiprocessing: bool = True,
        postprocessing_callbacks: list[Callable] | None = None,
        silent_errors: bool = False,
        **kwargs,
    ) -> list[dict[str, np.ndarray]]:
        """
        Generate trajectory ensembles for a given set of dynamical systems.
        """
        ensembles = []
        total_iterations = self.num_param_perturbations * self.num_ics
        pbar = tqdm(total=total_iterations, desc="Generating ensembles")

        with managed_cache(self.ic_sampler, use_multiprocessing):
            pp_rng_stream = self.rng.spawn(self.num_param_perturbations)
            for i, param_rng in enumerate(pp_rng_stream):
                param_perturbed_systems = self._init_perturbations(
                    systems,
                    param_rng=param_rng,
                    perturb_params=True,
                    use_multiprocessing=use_multiprocessing,
                )

                # filter out parameterless systems or
                # systems that failed to transform parameters for any reason
                excluded_pperts = [
                    sys if isinstance(sys, str) else sys.name  # type: ignore
                    for sys, pp_sys in zip(systems, param_perturbed_systems)
                    if pp_sys is None
                ]
                param_perturbed_systems = [
                    sys for sys in param_perturbed_systems if sys is not None
                ]

                if self.ic_sampler is not None and isinstance(
                    self.ic_sampler, OnAttractorInitCondSampler
                ):
                    self.ic_sampler.clear_cache()

                ic_rng_stream = param_rng.spawn(self.num_ics)
                for j, ic_rng in enumerate(ic_rng_stream):
                    sample_idx = i * len(ic_rng_stream) + j + 1

                    # after the parameter perturbation, perturb the initial conditions
                    ic_perturbed_systems = self._init_perturbations(
                        param_perturbed_systems,
                        ic_rng=ic_rng,
                        perturb_ics=True,
                        use_multiprocessing=use_multiprocessing,
                    )
                    excluded_systems = [
                        sys if isinstance(sys, str) else sys.name  # type: ignore
                        for sys, ic_sys in zip(systems, ic_perturbed_systems)
                        if ic_sys is None
                    ] + excluded_pperts  # systems that failed ic and param transforms
                    perturbed_systems = [
                        sys for sys in ic_perturbed_systems if sys is not None
                    ]
                    assert len(perturbed_systems) + len(excluded_systems) == len(
                        systems
                    )

                    ensemble = make_trajectory_ensemble(
                        self.num_points,
                        subset=perturbed_systems,
                        pts_per_period=self.num_points // self.num_periods,
                        event_fns=self.events,
                        use_multiprocessing=use_multiprocessing,
                        silent_errors=silent_errors,
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

                    pbar.update(1)
                    pbar.set_postfix({"param_idx": i, "ic_idx": j})

        return ensembles

    def _reset_events_callback(self, *args, **kwargs) -> None:
        for event in self.events or []:
            if hasattr(event, "reset") and callable(event.reset):
                event.reset()

    def save_failed_integrations_callback(self, sample_idx, ensemble, **kwargs):
        excluded_keys = kwargs.get("excluded_keys", [])
        if len(excluded_keys) > 0:
            logger.warning(f"Integration failed for {len(excluded_keys)} systems")
            for dyst_name in excluded_keys:
                self.failed_integrations[dyst_name].append(sample_idx)

    def _validate_and_save_ensemble_callback(
        self,
        num_total_samples: int,
        samples_process_interval: int,
        save_dyst_dir: str | None = None,
        failed_dyst_dir: str | None = None,
        save_params_dir: str | None = None,
        save_traj_stats_dir: str | None = None,
    ):
        """
        Callback to process and save ensembles and parameters
        """
        ensemble_list = []

        def _callback(sample_idx, ensemble, **kwargs):
            if len(ensemble.keys()) == 0:
                if save_dyst_dir is not None:
                    logger.warning("No successful trajectories for this sample")
                return

            ensemble_list.append(ensemble)

            is_last_sample = (sample_idx + 1) == num_total_samples
            if ((sample_idx + 1) % samples_process_interval) == 0 or is_last_sample:
                self._process_and_save_ensemble(
                    ensemble_list,
                    sample_idx,
                    perturbed_systems=kwargs.get("perturbed_systems"),
                    save_dyst_dir=save_dyst_dir,
                    failed_dyst_dir=failed_dyst_dir,
                    save_params_dir=save_params_dir,
                    save_traj_stats_dir=save_traj_stats_dir,
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
        save_traj_stats_dir: str | None = None,
    ) -> None:
        """
        Process the ensemble list by checking for valid attractors and filtering out invalid ones.
        Also, transposes and stacks trajectories to get shape (num_samples, num_dims, num_timesteps).
        """
        # stack and transpose to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
        ensemble_sys_names = [sys for ens in ensemble_list for sys in ens.keys()]
        ensemble = {
            sys: np.stack(
                [ens[sys] for ens in ensemble_list if sys in ens], axis=0
            ).transpose(0, 2, 1)
            for sys in ensemble_sys_names
        }

        # TEMPORARY: remove driver dims from trajectories
        if perturbed_systems is not None:
            dims = {sys.name: sys.driver_dim for sys in perturbed_systems}
            ensemble = {sys: traj[:, dims[sys] :, :] for sys, traj in ensemble.items()}

        if self.attractor_validator is not None:
            logger.info(f"Applying attractor validator to {len(ensemble)} systems")
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
                overwrite=True,  # idk it is what it is
                base_sample_idx=sample_idx,
            )

        if failed_dyst_dir is not None:
            process_trajs(
                failed_dyst_dir,
                failed_ensemble,
                split_coords=self.split_coords,
                verbose=self.verbose,
                overwrite=True,  # idk it is what it is
                base_sample_idx=sample_idx,
            )

        if save_params_dir is not None and perturbed_systems is not None:
            successful_systems = [
                sys for sys in perturbed_systems if sys.name in ensemble.keys()
            ]
            failed_systems = [
                sys for sys in perturbed_systems if sys.name in failed_ensemble.keys()
            ]

            success_dir = os.path.join(save_params_dir, "successes.json")
            self._save_parameters(sample_idx, successful_systems, success_dir)

            fail_dir = os.path.join(save_params_dir, "failures.json")
            self._save_parameters(sample_idx, failed_systems, fail_dir)

            # only save system stats for successful samples, and if we also save parameters
            if save_traj_stats_dir is not None:
                self._save_traj_stats(ensemble, save_dir=save_traj_stats_dir)

    def _save_parameters(
        self,
        sample_idx: int,
        perturbed_systems: list[BaseDyn],
        save_path: str | None = None,
    ) -> None:
        if save_path is None or len(perturbed_systems) == 0:
            return
        logger.info(f"Saving parameters to {save_path}")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                param_dict = json.load(f)
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            param_dict = {}

        for sys in perturbed_systems:
            if sys.name not in param_dict:
                param_dict[sys.name] = []

            if isinstance(sys, SkewProduct):
                serialized_params = {
                    "sample_idx": sample_idx,
                    "ic": sys.ic.tolist(),
                    "driver_params": dict_demote_from_numpy(sys.driver.params),
                    "response_params": dict_demote_from_numpy(sys.response.params),
                    "driver_dim": sys.driver_dim,
                    "response_dim": sys.response_dim,
                    "coupling_map": sys.coupling_map._serialize(),
                }
            else:
                serialized_params = {
                    "sample_idx": sample_idx,
                    "ic": sys.ic.tolist(),
                    "params": dict_demote_from_numpy(sys.params),
                    "dim": sys.dimension,
                }

            param_dict[sys.name].append(serialized_params)

        with open(save_path, "w") as f:
            json.dump(param_dict, f, indent=4)

    def _save_traj_stats(
        self,
        ensemble: dict[str, np.ndarray],
        save_dir: str | None = None,
    ) -> None:
        """
        Save trajectory statistics to a json file.
        We do this for downstream analysis and re-initialization without depending on loading trajectories from Arrow files
        TODO: pass in systems: List[DynSys] so we can also use it to save flow_rms
        """
        system_names = list(ensemble.keys())
        if save_dir is None or len(system_names) == 0:
            return
        save_path = os.path.join(save_dir, "traj_stats.json")
        logger.info(f"Saving trajectory stats to {save_path}")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                traj_stats = json.load(f)
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            traj_stats = {}

        for sys_name, trajectories in ensemble.items():
            # NOTE: while we can get ic, mean, and std from system attributes, the latter two are only computed if standardize=True
            if sys_name not in traj_stats:
                traj_stats[sys_name] = {"ic": [], "mean": [], "std": [], "mean_amp": []}

            init_conds, means, stds, mean_amps = [], [], [], []
            for _, traj in enumerate(trajectories):
                init_conds.append(traj[:, 0].tolist())
                means.append(traj.mean(axis=1).tolist())
                stds.append(traj.std(axis=1).tolist())
                mean_amps.append(np.mean(np.abs(traj), axis=1).tolist())

            traj_stats[sys_name]["ic"].extend(init_conds)
            traj_stats[sys_name]["mean"].extend(means)
            traj_stats[sys_name]["std"].extend(stds)
            traj_stats[sys_name]["mean_amp"].extend(mean_amps)

        with open(save_path, "w") as f:
            json.dump(traj_stats, f, indent=4)

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
