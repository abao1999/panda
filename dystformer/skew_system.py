"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

import dysts.flows as dfl
import numpy as np
from dysts.base import BaseDyn
from scipy.integrate import solve_ivp
from tqdm import trange

from dystformer.dyst_data import DystData
from dystformer.sampling import InstabilityEvent, TimeLimitEvent
from dystformer.utils import (
    construct_basic_affine_map,
    filter_dict,
    sample_index_pairs,
)


@dataclass
class SkewSystem:
    """
    A skew-product dynamical system, which is a pair of dynamical systems: a driver and a response.
    The driver and response are coupled together by a custom user-defined map.

    The class takes in two BaseDyn objects, which are dynamical systems.
    If no coupling map is provided, a basic affine map is constructed by default.
    """

    driver: BaseDyn
    response: BaseDyn
    coupling_map: Optional[np.ndarray] = None
    compute_coupling_strength: bool = False
    events: Optional[List] = None

    def __post_init__(self):
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

        self.n_driver, self.n_response = len(self.driver.ic), len(self.response.ic)
        self.dt = min(self.driver.dt, self.response.dt)

        # set integration time
        self.tlim = 20 * max(self.driver.period, self.response.period)
        self.tpts = np.linspace(0, self.tlim, 10_000)

        # set coupling strength kappa
        self.kappa = np.ones(self.n_response)  # dummy
        if self.compute_coupling_strength:
            self.kappa = self._compute_coupling()

        # set coupling map (default to basic affine map)
        if self.coupling_map is None:
            self.coupling_map = construct_basic_affine_map(
                self.n_driver, self.n_response, self.kappa
            )
        print("Coupling map: ", self.coupling_map)

    def _compute_coupling(self, ref_traj_len: int = 1000) -> np.ndarray:
        """
        Compute the coupling constants per dimension between the driver and response systems
        """
        print(
            f"Computing coupling strength between {self.driver.name} and {self.response.name}"
        )
        sol_driver = self.driver.make_trajectory(ref_traj_len)
        sol_response = self.response.make_trajectory(ref_traj_len)

        amp_driver = np.mean(np.abs(sol_driver), axis=0)
        amp_response = np.mean(np.abs(sol_response), axis=0)

        k = min(self.n_driver, self.n_response)
        kappa = amp_response[:k] / amp_driver[:k]
        return kappa

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

    def _get_skew_rhs(
        self,
        couple_phase_space: bool = True,
        couple_flows: bool = False,
    ) -> Callable:
        """
        Wrapper for skew-product system rhs, taking in a pre-computed affine map
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
            if couple_phase_space:
                _, _, x_response = self._apply_coupling_map(x_driver, x_response)

            # Compute the flow of the driver and response systems
            flow_driver = np.array(self.driver.rhs(x_driver, t))
            flow_response = np.array(self.response.rhs(x_response, t))

            # Method 2: couple flow vectors
            if couple_flows:
                flow_driver, _, flow_response = self._apply_coupling_map(
                    flow_driver, flow_response
                )

            skew_flow = np.concatenate([flow_driver, flow_response])
            return skew_flow

        return _skew_rhs

    def run(
        self, couple_phase_space: bool = True, couple_flows: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the skew-product system and return the trajectory of the driver and response systems
        """
        # combine initial conditions of driver and response system
        combined_ics = np.concatenate(
            [np.array(self.driver.ic), np.array(self.response.ic)]
        )

        # set up skew system rhs and solve
        skew_rhs = self._get_skew_rhs(
            couple_phase_space=couple_phase_space,  # couple x_driver, x_response
            couple_flows=couple_flows,  # couple flow_driver, flow_response
        )

        sol = solve_ivp(
            skew_rhs,
            [0, self.tlim],
            combined_ics,
            t_eval=self.tpts,
            first_step=self.dt,
            method="Radau",
            rtol=1e-6,
            atol=1e-6,
        )

        sol_driver = sol.y[: self.n_driver]
        sol_response = sol.y[self.n_driver : self.n_driver + self.n_response]
        return (sol_driver, sol_response)


@dataclass
class SkewEnsemble:
    """
    Generate an ensemble of skew-product dynamical systems, which are pairs of dynamical systems coupled together.
    """

    system_names: List[str]
    n_combos: int = 10
    compute_coupling_strength: bool = False
    events: Optional[List] = None

    def __post_init__(self):
        # get unique system names
        self.system_names = list(set(self.system_names))
        self.n_systems = len(self.system_names)

        random_pairs = sample_index_pairs(self.n_systems, self.n_combos)
        self.skew_pairs = [
            (self.system_names[i], self.system_names[j]) for i, j in random_pairs
        ]

    def _compute_skew_sol(
        self,
        driver_name: str,
        response_name: str,
        couple_phase_space: bool = True,
        couple_flows: bool = False,
    ) -> np.ndarray:
        """
        Compute the solution of a skew-product dynamical system (response system)
        """
        # NOTE: coupling map matrix is recomputed for each skew system
        skew_sys = SkewSystem(
            driver=getattr(dfl, driver_name)(),
            response=getattr(dfl, response_name)(),
            compute_coupling_strength=self.compute_coupling_strength,
            events=self.events,
        )
        _, sol_response = skew_sys.run(
            couple_phase_space=couple_phase_space,
            couple_flows=couple_flows,
        )
        return sol_response

    def multiprocess_generate_ensemble(
        self, couple_phase_space: bool = True, couple_flows: bool = False
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Generate an ensemble of skew-product dynamical systems, multiprocessed
        """
        with Pool() as pool:
            results = pool.starmap(
                self._compute_skew_sol,
                [
                    (driver_name, response_name, couple_phase_space, couple_flows)
                    for driver_name, response_name in self.skew_pairs
                ],
            )
        ensemble = {
            (driver_name, response_name): sol_response
            for (driver_name, response_name), sol_response in zip(
                self.skew_pairs, results
            )
        }
        return ensemble


@dataclass
class SkewData(DystData):
    """
    A dataset of skew-product dynamical systems
    """

    system_names: List[str]
    n_combos: int = 10
    compute_coupling_strength: bool = False
    couple_phase_space: bool = True
    couple_flows: bool = False

    def __post_init__(self):
        super().__post_init__()
        # get unique system names
        self.system_names = list(set(self.system_names))
        self.n_systems = len(self.system_names)

    def _generate_ensembles(
        self,
        system_names: List[str],
        postprocessing_callbacks: List[Callable] = [],
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        ensembles = []
        pp_rng_stream = np.random.default_rng(self.rseed).spawn(
            self.num_param_perturbations
        )
        for i, param_rng in zip(range(self.num_param_perturbations), pp_rng_stream):
            if self.param_sampler is not None:
                self.param_sampler.set_rng(param_rng)

            for j in trange(self.num_ics):
                sample_idx = i * self.num_ics + j

                # reset events that have a reset method
                self._reset_events()

                skew_ensemble_generator = SkewEnsemble(
                    system_names=self.system_names,
                    n_combos=self.n_combos,
                    compute_coupling_strength=self.compute_coupling_strength,
                    events=self.events,
                )

                ensemble = skew_ensemble_generator.multiprocess_generate_ensemble(
                    couple_phase_space=self.couple_phase_space,
                    couple_flows=self.couple_flows,
                )
                ensemble, excluded_keys = filter_dict(ensemble)
                ensembles.append(ensemble)

                for callback in postprocessing_callbacks:
                    callback(ensemble, excluded_keys, sample_idx)

        return ensembles
