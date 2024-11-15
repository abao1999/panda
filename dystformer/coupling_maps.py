"""
Coupling maps for skew systems.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class AdditiveCouplingMap:
    """
    Simple additive coupling map between driver and response flows
    """

    driver_dim: int
    response_dim: int

    driver_scale: float = 1
    response_scale: float = 1

    @property
    def dim(self) -> int:
        return min(self.driver_dim, self.response_dim)

    def transform_params(self, param_transform: Callable):
        self.driver_scale = param_transform("driver_scale", self.driver_scale)
        self.response_scale = param_transform("response_scale", self.response_scale)

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        return (
            self.driver_scale * driver[: self.dim]
            + self.response_scale * response[: self.dim]
        )

    def jac(
        self, driver: np.ndarray, response: np.ndarray, wrt: str = "driver"
    ) -> np.ndarray:
        if wrt == "driver":
            return np.eye(self.driver_dim)[: self.dim]
        elif wrt == "response":
            return np.eye(self.response_dim)[: self.dim]
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
