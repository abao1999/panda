"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from dysts.base import BaseDyn
from scipy.integrate import solve_ivp


@dataclass
class SkewSystem:
    driver: BaseDyn
    response: BaseDyn
    coupling_map: Optional[np.ndarray] = None
    adapt_coupling_strength: bool = False

    def __post_init__(self):
        self.n_driver, self.n_response = len(self.driver.ic), len(self.response.ic)
        self.dt = min(self.driver.dt, self.response.dt)

        # set integration time
        self.tlim = 20 * max(self.driver.period, self.response.period)
        self.tpts = np.linspace(0, self.tlim, 10_000)

        # set coupling strength kappa
        self.kappa = np.ones(self.n_response)  # dummy
        if self.adapt_coupling_strength:
            self.kappa = self._compute_coupling()

        # set coupling map (default to basic affine map)
        if self.coupling_map is None:
            self.coupling_map = construct_basic_affine_map(
                self.n_driver, self.n_response, self.kappa
            )
        print("Coupling map: ", self.coupling_map)

    def _compute_coupling(self) -> np.ndarray:
        """
        Compute the coupling constants per dimension between the driver and response systems
        TODO: after making dyst_data, we have a folder of trajectories, we can use this to compute coupling strength by reading from arrow file instead of generating trajectories
        """
        print(
            f"Computing coupling strength between {self.driver.name} and {self.response.name}"
        )
        k = min(self.n_driver, self.n_response)

        start_time = time.time()
        sol_driver = self.driver.make_trajectory(1000)
        sol_response = self.response.make_trajectory(1000)
        end_time = time.time()

        print(f"Trajectory generation time: {end_time - start_time} seconds")

        amp_driver = np.mean(np.abs(sol_driver), axis=0)
        amp_response = np.mean(np.abs(sol_response), axis=0)

        print("Amplitude of driver system: ", amp_driver)
        print("Amplitude of response system: ", amp_response)

        kappa = amp_response[:k] / amp_driver[:k]

        print("Coupling strength: ", kappa)
        return kappa

    def _apply_coupling_map(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """
        Apply a coupling map (e.g. affine map) to an augmented stacked vector of (x, y, 1)
        """
        n, m = len(x), len(y)
        # case of basic affine map: (x, y, 1) -> (x, y, x + y)
        transformed_vector = self.coupling_map @ np.hstack([x, y, 1])
        x = transformed_vector[:n]
        y = transformed_vector[n : n + m]
        return [x, y, transformed_vector[n + m :]]

    def _get_skew_rhs(
        self,
        couple_phase_space: bool = True,
        couple_flows: bool = False,
    ):
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

            # Method 1
            if couple_phase_space:
                _, _, x_response = self._apply_coupling_map(x_driver, x_response)

            # Compute the flow of the driver and response systems
            flow_driver = np.array(self.driver.rhs(x_driver, t))
            flow_response = np.array(self.response.rhs(x_response, t))

            # Method 2 - this seems like a weird thing to do
            if couple_flows:
                flow_driver, _, flow_response = self._apply_coupling_map(
                    flow_driver, flow_response
                )

            # # Method 3 - normalize flow rhs on the fly, both to unit norm
            # # kappa = np.linalg.norm(flow_response) / np.linalg.norm(flow_driver)
            # kappa_driver = 1 / np.linalg.norm(flow_driver)
            # kappa_response = 1 / np.linalg.norm(flow_response)
            # # affine_map = construct_basic_affine_map(n_driver, n_response, kappa)
            # flow_response = (kappa_driver * flow_driver) + (
            #     kappa_response * flow_response
            # )

            skew_flow = np.concatenate([flow_driver, flow_response])
            return skew_flow

        return _skew_rhs

    def run(self, couple_phase_space: bool = True, couple_flows: bool = False):
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

        start_time = time.time()
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
        end_time = time.time()
        print(f"Integration time: {end_time - start_time} seconds")

        sol_driver = sol.y[: self.n_driver]
        sol_response = sol.y[self.n_driver : self.n_driver + self.n_response]
        return [sol_driver, sol_response]


### Functions to make basic affine maps ###
def pad_array(arr: np.ndarray, n2: int, m2: int) -> np.ndarray:
    """
    General util to pad a 2D array to a target shape that is bigger than original shape
    """
    # Get the original shape (n1 x m1)
    n1, m1 = arr.shape
    # Calculate the padding amounts
    pad_rows = n2 - n1
    pad_cols = m2 - m1
    # Check if padding is needed
    if pad_rows < 0 or pad_cols < 0:
        raise ValueError(
            "Target dimensions must be greater than or equal to original dimensions."
        )
    # Apply padding: ((before_rows, after_rows), (before_cols, after_cols))
    padded_arr = np.pad(
        arr, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
    )
    return padded_arr


def construct_basic_affine_map(
    n: int,
    m: int,
    kappa: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """
    Construct an affine map that sends (x, y, 1) -> (x, y, x + y)
    where x and y have lengths n and m respectively, and n <= m

    Parameters:
    n: driver system dimension
    m: response system dimension

    Returns:
    A: the affine map matrix (2D array), block matrix (n + 2m) x (n + m + 1)
    """
    I_n = np.eye(n)  # n x n identity matrix
    I_m = np.eye(m)  # m x m identity matrix

    assert type(kappa) in [float, np.ndarray], "coupling strength kappa must be a float or a list of floats"  # type: ignore

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


# def test_map():
#     """
#     Simple test for affine map construction and application
#     """
#     x = np.array([1, 2, 3])
#     y = np.array([4, 5, 6, 7])
#     kappa = [1, 2, 3]
#     A = construct_basic_affine_map(len(x), len(y), kappa=kappa)
#     print(A)
#     res = apply_affine_map(A, x, y)
#     print(res)
#     print(np.concatenate(res))
