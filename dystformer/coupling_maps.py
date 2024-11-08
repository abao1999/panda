"""
Coupling maps for skew systems.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AdditiveCouplingMap:
    """
    Simple additive coupling map between driver and response flows
    """

    driver_dim: int
    response_dim: int

    driver_scale: float = 10
    response_scale: float = 1

    @property
    def dim(self) -> int:
        return min(self.driver_dim, self.response_dim)

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        return self.driver_scale * np.tanh(
            driver[: self.dim]
        ) + self.response_scale * np.tanh(response[: self.dim])

    def jac(
        self, driver: np.ndarray, response: np.ndarray, wrt: str = "driver"
    ) -> np.ndarray:
        if wrt == "driver":
            d_tanh = self.driver_scale * (
                1 - np.tanh(self.driver_scale * driver[: self.dim]) ** 2
            )
            return np.eye(self.driver_dim)[: self.dim] * d_tanh
        elif wrt == "response":
            d_tanh = self.response_scale * (
                1 - np.tanh(self.response_scale * response[: self.dim]) ** 2
            )
            return np.eye(self.response_dim)[: self.dim] * d_tanh
        else:
            raise ValueError(f"Invalid wrt argument: {wrt}")


@dataclass
class RandomLinearCouplingMap:
    """
    Affine coupling map between driver and response flows

    NOTE: for now, assume embedding dimension is the min of driver and response dimensions
    """

    driver_dim: int
    response_dim: int

    random_seed: int = 0
    coupling_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.coupling_matrix is None:
            self.rng = np.random.default_rng(self.random_seed)
            self.coupling_matrix = self.rng.normal(
                size=(self.dim, self.driver_dim + self.response_dim)
            ) / (self.driver_dim + self.response_dim)
        else:
            assert self.coupling_matrix.shape == (
                self.dim,
                self.driver_dim + self.response_dim,
            )

    @property
    def dim(self) -> int:
        return min(self.driver_dim, self.response_dim)

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        return self.coupling_matrix @ np.hstack([driver, response])

    def jac(self, driver: np.ndarray, response: np.ndarray, wrt: str = "driver"):
        if wrt == "driver":
            return self.coupling_matrix[:, : self.driver_dim]
        elif wrt == "response":
            return self.coupling_matrix[:, self.driver_dim :]
        else:
            raise ValueError(f"Invalid wrt argument: {wrt}")
