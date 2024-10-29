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

    @property
    def dim(self) -> int:
        return min(self.driver_dim, self.response_dim)

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        return driver[: self.dim] + response[: self.dim]


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
            )
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
