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
class AdditiveCouplingMap(BaseCouplingMap):
    """
    Simple additive coupling map between driver and response flows
    """

    driver_scale: float = 1
    response_scale: float = 1

    @property
    def dim(self) -> int:
        return min(self.driver_dim, self.response_dim)

    @property
    def n_params(self) -> int:
        return 2

    def transform_params(self, param_transform: Callable) -> bool:
        self.driver_scale = param_transform("driver_scale", self.driver_scale)
        self.response_scale = param_transform("response_scale", self.response_scale)
        return True

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
class RandomAdditiveCouplingMap(BaseCouplingMap):
    """
    Random additive coupling map which samples indices from driver and response flows
    """

    driver_scale: float = 1
    response_scale: float = 1

    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        self.indices = self.rng.choice(self.dim, size=self.dim, replace=False)

    @property
    def dim(self) -> int:
        return min(self.driver_dim, self.response_dim)

    @property
    def n_params(self) -> int:
        return 2 + self.indices.size

    def transform_params(self, param_transform: Callable) -> bool:
        self.driver_scale = param_transform("driver_scale", self.driver_scale)
        self.response_scale = param_transform("response_scale", self.response_scale)
        self.indices = self.rng.choice(self.dim, size=self.dim, replace=False)
        return True

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        return (
            self.driver_scale * driver[self.indices]
            + self.response_scale * response[self.indices]
        )

    def jac(
        self, driver: np.ndarray, response: np.ndarray, wrt: str = "driver"
    ) -> np.ndarray:
        if wrt == "driver":
            return np.eye(self.driver_dim)[self.indices]
        elif wrt == "response":
            return np.eye(self.response_dim)[self.indices]
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
            return self.coupling_matrix[:, : self.driver_dim]  # type: ignore
        elif wrt == "response":
            return self.coupling_matrix[:, self.driver_dim :]  # type: ignore
        else:
            raise ValueError(f"Invalid wrt argument: {wrt}")
