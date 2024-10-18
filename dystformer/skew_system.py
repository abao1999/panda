"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import dysts.flows as dfl
import numpy as np
from dysts.base import BaseDyn
from gluonts.dataset.common import FileDataset
from scipy.integrate import solve_ivp

from dystformer.utils import (
    get_system_filepaths,
    sample_index_pairs,
    stack_and_extract_metadata,
)


@dataclass
class SkewEnsemble:
    """
    A collection of skew-product dynamical systems, which are pairs of dynamical systems coupled together.
    """

    system_names: List[str]
    n_combos: int = 1000
    compute_coupling_strength: bool = False
    source_data_dir: Optional[str] = None  # directory to source data for skew pairs
    source_split: str = "train"  # split of data to use for source data

    def __post_init__(self):
        # get unique system names
        self.system_names = list(set(self.system_names))
        self.n_systems = len(self.system_names)

        random_pairs = sample_index_pairs(self.n_systems, self.n_combos)
        self.skew_pairs = [
            (self.system_names[i], self.system_names[j]) for i, j in random_pairs
        ]

        # NOTE: coupling map matrix is recomputed for each skew system
        self.skew_systems = [
            SkewSystem(
                driver=getattr(dfl, driver_name)(),
                response=getattr(dfl, response_name)(),
                compute_coupling_strength=self.compute_coupling_strength,
                source_data_dir=self.source_data_dir,
                source_split=self.source_split,
            )
            for driver_name, response_name in self.skew_pairs
        ]
        # TODO: incorporate parameter and IC samplers
        # TODO: maintain a cache of computed amplitudes from trajectory data,
        # so we don't recompute them every time we make a new skew system

    def generate_ensemble(
        self, couple_phase_space: bool = True, couple_flows: bool = False
    ) -> Dict[Tuple[str, str], List[np.ndarray]]:
        ensemble = defaultdict(list)
        for skew_sys in self.skew_systems:
            _, sol_response = skew_sys.run(
                couple_phase_space=couple_phase_space, couple_flows=couple_flows
            )
            skew_pair = (skew_sys.driver.name, skew_sys.response.name)
            ensemble[skew_pair].append(sol_response)

        return ensemble

    def save_ensemble(self, save_dir: str):
        pass


@dataclass
class SkewSystem:
    """
    A skew-product dynamical system, which is a pair of dynamical systems: a driver and a response.
    The driver and response are coupled together by a custom user-defined map.

    The class takes in two BaseDyn objects, which are dynamical systems.
    It optionally takes in a source data directory to load trajectory data from,
    otherwise it will generate trajectories on the fly to compute the coupling strength.

    If no coupling map is provided, a basic affine map is constructed by default.
    """

    driver: BaseDyn
    response: BaseDyn
    coupling_map: Optional[np.ndarray] = None
    compute_coupling_strength: bool = False
    source_data_dir: Optional[str] = None  # directory to source data for skew pairs
    source_split: str = "train"  # split of data to use for source data

    def __post_init__(self):
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

    def _load_dyst_traj(
        self,
        dyst_name: str,
        sample_idx: Optional[int] = None,
        one_dim_target: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Try to load dyst trajectory from self.source_data_dir. If that fails, return None.
        """
        if self.source_data_dir is None:
            return None

        filepaths = []
        try:
            filepaths = get_system_filepaths(
                dyst_name, self.source_data_dir, self.source_split
            )
        except Exception as e:
            print(f"Failed to load {dyst_name} from {self.source_data_dir}: {e}")
            return None

        if len(filepaths) == 0:
            return None

        sample_idx = sample_idx or np.random.randint(len(filepaths))
        filepath = filepaths[sample_idx]
        gts_dataset = FileDataset(
            path=Path(filepath),
            freq="h",
            one_dim_target=one_dim_target,  # False for PatchTST
        )
        # extract the coordinates
        dyst_coords, _ = stack_and_extract_metadata(gts_dataset)

        return dyst_coords

    def _compute_coupling(self, ref_traj_len: int = 1000) -> np.ndarray:
        """
        Compute the coupling constants per dimension between the driver and response systems
        TODO: after making dyst_data, we have a folder of trajectories, we can use this to compute coupling strength by reading from arrow file instead of generating trajectories
        """
        print(
            f"Computing coupling strength between {self.driver.name} and {self.response.name}"
        )
        sol_driver = self._load_dyst_traj(self.driver.name)
        sol_response = self._load_dyst_traj(self.response.name)

        # if we can't load the data, generate it
        if sol_driver is None:
            sol_driver = self.driver.make_trajectory(ref_traj_len)
        if sol_response is None:
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


### Functions to make basic affine maps ###
def pad_array(arr: np.ndarray, n2: int, m2: int) -> np.ndarray:
    """
    Pad an array to a target shape that is bigger than original shape
    """
    n1, m1 = arr.shape
    pad_rows, pad_cols = n2 - n1, m2 - m1
    if pad_rows < 0 or pad_cols < 0:
        raise ValueError(
            "Target dimensions must be greater than or equal to original dimensions."
        )
    return np.pad(
        arr, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
    )


def construct_basic_affine_map(
    n: int,
    m: int,
    kappa: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """
    Construct an affine map that sends (x, y, 1) -> (x, y, x + y)
    where x and y have lengths n and m respectively, and n <= m
    Args:
        n: driver system dimension
        m: response system dimension
        kappa: coupling strength, either a float or a list of floats
    Returns:
        A: the affine map matrix (2D array), block matrix (n + 2m) x (n + m + 1)
    """
    I_n = np.eye(n)  # n x n identity matrix
    I_m = np.eye(m)  # m x m identity matrix

    assert isinstance(
        kappa, (float, np.ndarray)
    ), "coupling strength kappa must be a float or a list of floats"

    if isinstance(kappa, float):
        bottom_block = np.hstack(
            [kappa * pad_array(I_n if n < m else I_m, m, n), I_m, np.zeros((m, 1))]
        )
    else:  # kappa is a list of floats
        k = min(n, m)
        assert len(kappa) == k, "coupling strength kappa must be of length min(n, m)"  # type: ignore
        bottom_block = np.hstack(
            [pad_array(np.diag(kappa), m, n), I_m, np.zeros((m, 1))]
        )

    top_block = np.hstack([I_n, np.zeros((n, m)), np.zeros((n, 1))])
    middle_block = np.hstack([np.zeros((m, n)), I_m, np.zeros((m, 1))])

    A = np.vstack([top_block, middle_block, bottom_block])
    return A
