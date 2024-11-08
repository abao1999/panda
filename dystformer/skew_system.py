"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import logging
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

import dysts.flows as dfl
import numpy as np
from dysts.base import DynSys, staticjit
from scipy.integrate import solve_ivp
from tqdm import trange

from dystformer.coupling_maps import AdditiveCouplingMap
from dystformer.dyst_data import DystData
from dystformer.sampling import InstabilityEvent, TimeLimitEvent
from dystformer.utils import (
    construct_basic_affine_map,
    filter_dict,
    sample_index_pairs,
)

logger = logging.getLogger(__name__)


class SkewProduct(DynSys):
    def __init__(
        self,
        driver: DynSys,
        response: DynSys,
        coupling_map: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        **kwargs,
    ):
        super().__init__(
            metadata_path=None,
            parameters={},  # dummy: parameters are handled in the overwritten methods below
            metadata={
                "name": f"{driver.name}_{response.name}",
                "dimension": driver.dimension + response.dimension,
            },
            **kwargs,
        )
        self.driver = driver
        self.response = response

        self.driver_dim, self.response_dim = driver.dimension, response.dimension

        assert hasattr(driver, "ic") and hasattr(
            response, "ic"
        ), "Driver and response must have default initial conditions"
        self.ic = np.concatenate([self.driver.ic, self.response.ic])

        self.mean = np.concatenate([self.driver.mean, self.response.mean])
        self.std = np.concatenate([self.driver.std, self.response.std])

        # default to additively forcing the response with the driver
        if coupling_map is None:
            self.coupling_map = AdditiveCouplingMap(self.driver_dim, self.response_dim)
        else:
            self.coupling_map = coupling_map

    def transform_ic(self, ic_transform: Callable):
        self.driver.transform_ic(ic_transform)
        self.response.transform_ic(ic_transform)
        self.ic = np.concatenate([self.driver.ic, self.response.ic])

    def transform_params(self, param_transform: Callable):
        self.driver.transform_params(param_transform)
        self.response.transform_params(param_transform)

    def has_jacobian(self):
        return self.driver.has_jacobian() and self.response.has_jacobian()

    def rhs(self, X, t):
        driver, response = X[: self.driver_dim], X[self.driver_dim :]
        driver_rhs = np.asarray(self.driver.rhs(driver, t))
        response_rhs = np.asarray(self.response.rhs(response, t))
        coupled_rhs = self.coupling_map(driver_rhs, response_rhs)
        print(
            f"driver_rhs: {np.linalg.norm(driver_rhs)}, response_rhs: {np.linalg.norm(response_rhs)}, coupled_rhs: {np.linalg.norm(coupled_rhs)}"
        )
        return np.concatenate([driver_rhs, coupled_rhs])

    def jac(self, X, t):
        driver, response = X[: self.driver_dim], X[self.driver_dim :]

        driver_jac = np.asarray(self.driver.jac(driver, t))
        coupling_jac_driver = self.coupling_map.jac(driver, response, wrt="driver")

        response_jac = np.asarray(self.response.jac(response, t))
        coupling_jac_response = self.coupling_map.jac(driver, response, wrt="response")

        return np.block(
            [
                [driver_jac, np.zeros((self.driver_dim, self.response_dim))],
                [
                    coupling_jac_driver @ driver_jac,
                    coupling_jac_response @ response_jac,
                ],
            ]
        )

    def __call__(self, X, t):
        return self.rhs(X, t)


@dataclass
class SkewSystem:
    """
    A skew-product dynamical system, which is a pair of dynamical systems: a driver and a response.
    The driver and response are coupled together by a custom user-defined map.

    The class takes in two BaseDyn objects, which are dynamical systems, class defined in the dysts library.
    If no coupling map is provided, a basic affine map is constructed by default.

    Args:
        driver (DynSys): The driver dynamical system.
        response (DynSys): The response dynamical system.
        coupling_map (Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]):
            A function that defines the coupling between driver and response systems.
            If None, a basic affine map is used. Default is None.
        couple_phase_space (bool): If True, couples the phase space of driver and response.
            Default is False.
        couple_flows (bool): If True, couples the flow vectors of driver and response.
            Default is True.
        events (Optional[List[Callable]]): List of event functions for numerical integration.
            If None, default events are used. Default is None.
    """

    driver: DynSys
    response: DynSys
    coupling_map: Optional[np.ndarray] = None
    couple_phase_space: bool = False
    couple_flows: bool = True
    events: Optional[List[Callable]] = None

    def __post_init__(self):
        self.name = f"{self.driver.name}_{self.response.name}"

        # Set default integration events if none are provided
        if self.events is None:
            warnings.warn(
                "No events provided for numerical integration. Defaulting to TimeLimitEvent with max_duration of 5 minutes \
                and InstabilityEvent with threshold of 1e4."
            )
            # events for solve_ivp
            time_limit_event = TimeLimitEvent(max_duration=60 * 5)  # 5 min time limit
            instability_event = InstabilityEvent(threshold=1e4)
            self.events = [time_limit_event, instability_event]

        # dimension of driver and response systems
        self.n_driver, self.n_response = len(self.driver.ic), len(self.response.ic)
        self.k = min(self.n_driver, self.n_response)

        # set integration time
        self.dt = min(self.driver.dt, self.response.dt)
        self.period = max(self.driver.period, self.response.period)

    def _compute_coupling_strength(self, ref_traj_len: int = 1000) -> np.ndarray:
        """
        Compute the coupling constants per dimension between the driver and response systems
        """
        # This is possibly not integrable, so we need to check if the trajectory is complete
        sol_driver = self.driver.make_trajectory(
            ref_traj_len, events=self.events, standardize=False
        )
        sol_response = self.response.make_trajectory(
            ref_traj_len, events=self.events, standardize=False
        )

        if sol_driver is None or sol_response is None:
            warnings.warn(
                f"Could not make a complete trajectory for {self.driver.name} and {self.response.name} with {ref_traj_len} points."
            )
            return np.ones(self.k)

        # compute amplitude of driver and response systems
        amp_driver = np.mean(np.abs(sol_driver), axis=0)  # type: ignore
        amp_response = np.mean(np.abs(sol_response), axis=0)  # type: ignore

        # compute coupling strength vector as ratio of amplitudes
        kappa = amp_response[: self.k] / amp_driver[: self.k]
        return kappa

    def _make_default_coupling_map(self):
        """
        Set the coupling map to be a basic affine map for the coupling between the driver and response systems
        """
        kappa = self._compute_coupling_strength()
        # kappa = np.ones(self.k)  # dummy
        self.coupling_map = construct_basic_affine_map(
            self.n_driver, self.n_response, kappa
        )

    def _apply_coupling_map(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """
        Apply a coupling map (e.g. affine map) to an augmented stacked vector of (x, y, 1)
            Case of basic affine map: (x, y, 1) -> (x, y, x + y)
        """
        n, m = len(x), len(y)
        transformed_vector = self.coupling_map @ np.hstack([x, y, 1])
        x = transformed_vector[:n]
        y = transformed_vector[n : n + m]
        return [x, y, transformed_vector[n + m :]]

    @staticjit
    def _rhs():
        pass

    def _get_skew_rhs(
        self,
    ) -> Callable:
        """
        Wrapper for skew-product system rhs, taking in a pre-computed affine map
        NOTE: we made coupling_map a class attribute, but consider passing it in as an arg here instead
        """

        def _skew_rhs(t, combined_ics):
            """
            Skew-product system rhs signature
            """
            # Split the combined initial conditions into driver and response systems
            x_driver, x_response = (
                combined_ics[: self.n_driver],
                combined_ics[self.n_driver : self.n_driver + self.n_response],
            )

            # Method 1: couple phase space
            if self.couple_phase_space:
                _, _, x_response = self._apply_coupling_map(x_driver, x_response)

            # Compute the flow of the driver and response systems
            flow_driver = np.array(self.driver.rhs(x_driver, t))
            flow_response = np.array(self.response.rhs(x_response, t))

            # Method 2: couple flow vectors
            if self.couple_flows:
                flow_driver, _, flow_response = self._apply_coupling_map(
                    flow_driver, flow_response
                )

            skew_flow = np.concatenate([flow_driver, flow_response])
            return skew_flow

        return _skew_rhs

    def rhs(self, t, X):
        """The right hand side of the skew system"""
        skew_rhs = self._get_skew_rhs()
        return skew_rhs(t, X)

    def __call__(self, t, X):
        """Wrapper around right hand side"""
        return self.rhs(t, X)

    def run(
        self,
        num_periods: int = 20,
        num_points: int = 10_000,
        method: str = "Radau",
        rtol: float = 1e-6,
        atol: float = 1e-6,
        standardize: bool = True,
        **kwargs,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run the skew-product system and return the trajectory of the driver and response systems
        TODO: adapt timescale?

        Args:
            num_periods: Number of periods to integrate over
            num_points: Number of points to integrate over
            method: Integration method
            rtol: Relative tolerance for integration
            atol: Absolute tolerance for integration
            standardize: Whether to standardize the solution to have mean 0 and std 1

        Currently, the integration time is set to be the Fourier timescale, which makes tlim = num_periods * period
        """

        # set coupling map (default to basic affine map)
        if self.coupling_map is None:
            self._make_default_coupling_map()

        # combine initial conditions of driver and response system
        combined_ics = np.concatenate(
            [np.array(self.driver.ic), np.array(self.response.ic)]
        )

        # standardize the rhs
        if standardize:
            warnings.warn(
                "Standardization not yet fully implemented and tested for skew systems"
            )

        # TODO: is the right standardization? i.e. can we just consider the driver and response systems separately?
        mu_driver = self.driver.mean if standardize else np.zeros(self.n_driver)
        std_driver = self.driver.std if standardize else np.ones(self.n_driver)
        mu_response = self.response.mean if standardize else np.zeros(self.n_response)
        std_response = self.response.std if standardize else np.ones(self.n_response)

        mu = np.concatenate([mu_driver, mu_response])
        std = np.concatenate([std_driver, std_response])

        def standard_rhs(t, X):
            # TODO: divide by zero error
            return self(t, X * std + mu) / std

        # This is the Fourier timescale
        tlim = num_periods * self.period
        tpts = np.linspace(0, tlim, num_points)

        if not np.isfinite(standard_rhs(0, combined_ics)).all():
            warnings.warn(
                f"Initial state of {self.driver.name} and {self.response.name} is invalid!"
            )
            return (None, None)

        sol = solve_ivp(
            standard_rhs,
            [0, tlim],
            (combined_ics - mu) / std,
            t_eval=tpts,
            first_step=self.dt,
            method=method,
            rtol=rtol,
            atol=atol,
            events=self.events,
            **kwargs,
        )

        is_complete_traj = sol.y.shape[-1] == num_points
        if not is_complete_traj:
            warnings.warn(
                f"Could not make a complete trajectory for {self.driver.name} and {self.response.name} with {num_periods} periods."
            )
            return (None, None)

        sol_driver = sol.y[: self.n_driver]
        sol_response = sol.y[self.n_driver : self.n_driver + self.n_response]
        return (sol_driver, sol_response)


@dataclass
class SkewEnsemble:
    """
    Generate an ensemble of skew-product dynamical systems, which are pairs of dynamical systems coupled together.
    Args:
        skew_pair_names: List of names of skew dynamical systems. The names are joined together by "_" to form the keys of the ensemble dictionary as strings.
            e.g. ["Lorenz_Rossler", "Rossler_Lorenz"] means two skew systems: Lorenz driving Rossler and Rossler driven by Lorenz.
        couple_phase_space: Whether to couple the phase space of the driver and response systems
        couple_flows: Whether to couple the flow vectors of the driver and response systems
        events: List of events for numerical integration
    """

    skew_pair_names: List[str]
    couple_phase_space: bool = True
    couple_flows: bool = False
    events: Optional[List[Callable]] = None

    def _compute_skew_sol(
        self,
        skew_name: str,
        num_periods: int = 20,
        num_points: int = 10_000,
        kwargs: Dict = {},
        ic_transform: Optional[Callable] = None,
        param_transform: Optional[Callable] = None,
        ic_rng: Optional[np.random.Generator] = None,
        param_rng: Optional[np.random.Generator] = None,
    ) -> Optional[np.ndarray]:
        """
        A helper function for multiprocessing.
        Compute the solution of a skew-product dynamical system (response system)
        Following the dysts make_trajectory convention, return None if the system cannot be made
            i.e. if the driver or response system is not found, return None

        We can compute the coupling strength using the trajectory cache of the initial condition transform
        """

        driver_name, response_name = skew_name.split("_")
        driver_sys = getattr(dfl, driver_name)()
        response_sys = getattr(dfl, response_name)()

        if param_transform is not None:
            if param_rng is not None:
                param_transform.set_rng(param_rng)
            # transform both the driver and response systems parameters
            driver_sys.transform_params(param_transform)
            response_sys.transform_params(param_transform)

        # the initial condition transform must come after the parameter transform
        # because suitable initial conditions may depend on the parameters
        if ic_transform is not None:
            if ic_rng is not None:
                ic_transform.set_rng(ic_rng)
            # transform both the driver and response systems initial conditions
            driver_sys.transform_ic(ic_transform)
            response_sys.transform_ic(ic_transform)

        try:
            skew_sys = SkewSystem(
                driver=driver_sys,
                response=response_sys,
                coupling_map=None,  # creates basic affine map by default
                couple_phase_space=self.couple_phase_space,
                couple_flows=self.couple_flows,
                events=self.events,
            )
        except AttributeError as e:
            logger.info(f"System {skew_name} could not be made. {e} Skipping...")
            return None

        # Avoid having to recompute coupling strength if ic sampler's trajectory cache is available
        can_use_traj_cache = ic_transform is not None and hasattr(
            ic_transform, "trajectory_cache"
        )
        # This sidesteps needing to recompute kappa and make default coupling map in SkewSystem run() method
        if can_use_traj_cache:
            traj_cache = ic_transform.trajectory_cache  # type: ignore
            if driver_name in traj_cache and response_name in traj_cache:
                traj_driver = traj_cache[driver_name]
                traj_response = traj_cache[response_name]

                amp_driver = np.mean(np.abs(traj_driver), axis=0)
                amp_response = np.mean(np.abs(traj_response), axis=0)

                n_driver, n_response = len(driver_sys.ic), len(response_sys.ic)
                k = min(n_driver, n_response)
                kappa = amp_response[:k] / amp_driver[:k]
                # Set coupling map
                skew_sys.coupling_map = construct_basic_affine_map(
                    n_driver, n_response, kappa
                )

        try:
            _, sol_response = skew_sys.run(
                num_periods=num_periods,
                num_points=num_points,
                **kwargs,
            )
        except Exception as e:
            logger.info(
                f"Error in skew system {skew_name} ({driver_name} driving {response_name}): {e}"
            )
            logger.info(f"Driver IC: {driver_sys.ic} | Response IC: {response_sys.ic}")
            logger.info(
                f"Driver params: {driver_sys.params} | Response params: {response_sys.params}"
            )
            return None

        # Transpose solution to match the structure of dysts make_trajectory_ensemble
        return sol_response.T if sol_response is not None else None

    def multiprocess_generate_ensemble(
        self,
        num_periods: int = 20,
        num_points: int = 10_000,
        ic_transform: Optional[Callable] = None,
        param_transform: Optional[Callable] = None,
        ic_rng: Optional[np.random.Generator] = None,
        param_rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Generate an ensemble of skew-product dynamical systems, multiprocessed
        """

        n_skew_systems = len(self.skew_pair_names)
        ic_rng_stream = (
            ic_rng.spawn(n_skew_systems) if ic_rng else [None] * n_skew_systems
        )
        param_rng_stream = (
            param_rng.spawn(n_skew_systems) if param_rng else [None] * n_skew_systems
        )

        with Pool() as pool:
            results = pool.starmap(
                self._compute_skew_sol,
                [
                    (
                        skew_name,
                        num_periods,
                        num_points,
                        kwargs,
                        ic_transform,
                        param_transform,
                        ic_rng,
                        param_rng,
                    )
                    for skew_name, param_rng, ic_rng in zip(
                        self.skew_pair_names, param_rng_stream, ic_rng_stream
                    )
                ],
            )
        ensemble = {
            skew_name: sol_response
            for skew_name, sol_response in zip(self.skew_pair_names, results)
        }
        return ensemble


@dataclass
class SkewData(DystData):
    """
    A dataset of skew-product dynamical systems, inherits from DystData
    Args:
        couple_phase_space: Whether to couple the phase space of the driver and response systems
        couple_flows: Whether to couple the flow vectors of the driver and response systems
    """

    couple_phase_space: bool = True
    couple_flows: bool = False

    def __post_init__(self):
        super().__post_init__()

    def sample_skew_pairs(
        self,
        system_names: List[str],
        n_combos: int = 10,
    ) -> List[str]:
        """
        Sample a list of unique skew-product dynamical system pairs
        """
        rng = np.random.default_rng(self.rseed)
        system_names = list(set(system_names))
        n_systems = len(system_names)
        random_pairs = sample_index_pairs(n_systems, n_combos, rng=rng)
        skew_pairs = [(system_names[i], system_names[j]) for i, j in random_pairs]
        skew_pair_names = [f"{driver}_{response}" for driver, response in skew_pairs]
        return skew_pair_names

    def _generate_ensembles(
        self,
        dysts_names: List[str],
        postprocessing_callbacks: List[Callable] = [],
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate ensembles of skew-product dynamical systems
        Args:
            dysts_names: List of names of skew dynamical systems. The names are joined together by "_" to form the keys of the ensemble dictionary as strings.
                e.g. ["Lorenz_Rossler", "Rossler_Lorenz"] means two skew systems: Lorenz driving Rossler and Rossler driven by Lorenz.
            postprocessing_callbacks: List of functions that perform postprocessing on the ensemble
        """
        ensembles = []
        pp_rng_stream = np.random.default_rng(self.rseed).spawn(
            self.num_param_perturbations
        )
        for i, param_rng in zip(range(self.num_param_perturbations), pp_rng_stream):
            if self.param_sampler is not None:
                self.param_sampler.set_rng(param_rng)

            if self.ic_sampler is not None:
                self.ic_sampler.clear_cache()  # type: ignore

            for j in trange(self.num_ics):
                sample_idx = i * self.num_ics + j

                # reset events that have a reset method
                self._reset_events()

                skew_ensemble_generator = SkewEnsemble(
                    skew_pair_names=dysts_names,
                    couple_phase_space=self.couple_phase_space,
                    couple_flows=self.couple_flows,
                    events=self.events,
                )

                ensemble = skew_ensemble_generator.multiprocess_generate_ensemble(
                    num_periods=self.num_periods,
                    num_points=self.num_points,
                    ic_transform=self.ic_sampler if sample_idx > 0 else None,  # type: ignore
                    param_transform=self.param_sampler if i > 0 else None,  # type: ignore
                    ic_rng=param_rng,
                    **kwargs,
                )
                ensemble, excluded_keys = filter_dict(ensemble)
                ensembles.append(ensemble)

                for callback in postprocessing_callbacks:
                    callback(ensemble, excluded_keys, sample_idx)

        return ensembles
