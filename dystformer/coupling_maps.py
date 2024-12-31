"""
Coupling maps for skew systems.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BaseCouplingMap:
    """Base class for coupling maps"""

    driver_dim: int
    response_dim: int

    @property
    def dim(self) -> int:
        raise NotImplementedError


@dataclass
class RandomAdditiveCouplingMap(BaseCouplingMap):
    """
    Simple additive coupling map between driver and response flows
    Optionally, randomly samples the indices of the driver and response flows to couple.

    NOTE: the dimensions are fixed to the response system
    """

    driver_scale: float = 1
    response_scale: float = 1

    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.random_seed is not None:
            self.rng = np.random.default_rng(self.random_seed)
            self.driver_indices = self.rng.choice(
                max(self.driver_dim, self.response_dim),
                self.response_dim,
                replace=False,
            )
        else:
            self.driver_indices = np.arange(self.response_dim)

    @property
    def n_params(self) -> int:
        # 2 params for the scales + self.response_dim params for the indices
        return 2 + (self.response_dim) * (self.random_seed is not None)

    def transform_params(self, param_transform: Callable) -> bool:
        self.driver_scale = param_transform("driver_scale", self.driver_scale)
        self.response_scale = param_transform("response_scale", self.response_scale)

        if self.random_seed is not None:
            self.driver_indices = self.rng.choice(
                max(self.driver_dim, self.response_dim),
                self.response_dim,
                replace=False,
            )

        return True

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        padded_driver = np.pad(driver, (0, max(self.response_dim - self.driver_dim, 0)))
        return (
            self.driver_scale * padded_driver[self.driver_indices]
            + self.response_scale * response
        )

    def jac(
        self, driver: np.ndarray, response: np.ndarray, wrt: str = "driver"
    ) -> np.ndarray:
        if wrt == "driver":
            djac = np.pad(
                np.eye(self.driver_dim),
                ((0, max(self.response_dim - self.driver_dim, 0)), (0, 0)),
            )
            return self.driver_scale * djac[self.driver_indices]
        elif wrt == "response":
            rjac = np.eye(self.response_dim)
            return self.response_scale * rjac
        else:
            raise ValueError(f"Invalid wrt argument: {wrt}")


@dataclass
class RandomLinearCouplingMap(BaseCouplingMap):
    """
    Affine coupling map between driver and response flows
    """

    random_seed: int = 0
    coupling_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.coupling_matrix is None:
            self.rng = np.random.default_rng(self.random_seed)
            self.coupling_matrix = self.rng.normal(
                size=(self.response_dim, self.driver_dim + self.response_dim)
            ) / (self.driver_dim + self.response_dim)
        else:
            assert self.coupling_matrix.shape == (
                self.response_dim,
                self.driver_dim + self.response_dim,
            )

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        return self.coupling_matrix @ np.hstack([driver, response])

    def jac(self, driver: np.ndarray, response: np.ndarray, wrt: str = "driver"):
        if wrt == "driver":
            return self.coupling_matrix[:, : self.driver_dim]  # type: ignore
        elif wrt == "response":
            return self.coupling_matrix[:, self.driver_dim :]  # type: ignore
        else:
            raise ValueError(f"Invalid wrt argument: {wrt}")
