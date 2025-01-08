"""
Coupling maps for skew systems.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

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

    driver_stats: dict[str, float] = field(default_factory=dict)
    response_stats: dict[str, float] = field(default_factory=dict)

    perturb_stats: bool = False

    randomize_driver_indices: bool = True
    normalization_strategy: Literal["Amplitude", "RMS"] | None = None

    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.random_seed is not None:
            self.rng = np.random.default_rng(self.random_seed)

        self.driver_indices = np.arange(self.response_dim)
        self.c_driver = self.c_response = 1.0
        if self.normalization_strategy is not None:
            self.c_driver, self.c_response = self._compute_normalization_constants()

    @property
    def n_params(self) -> int:
        # 2 params for the scales + self.response_dim params for the indices
        return 2 + (self.response_dim) * (
            self.random_seed is not None and self.randomize_driver_indices
        )

    def _compute_normalization_constants(self) -> tuple[float, float]:
        def get_stats(key_name: str) -> tuple[float, float]:
            if key_name not in self.driver_stats:
                raise ValueError(f"key '{key_name}' not found in driver stats")
            if key_name not in self.response_stats:
                raise ValueError(f"key '{key_name}' not found in response stats")
            return self.driver_stats[key_name], self.response_stats[key_name]

        if self.normalization_strategy == "Amplitude":
            amp_driver, amp_response = get_stats("amplitude")
            c_driver = amp_response / amp_driver
            c_response = 1.0
        elif self.normalization_strategy == "RMS":
            flow_rms_driver, flow_rms_response = get_stats("flow_rms")
            c_driver = 1 / flow_rms_driver
            c_response = 1 / flow_rms_response
        else:
            raise ValueError(
                f"Invalid normalization strategy: {self.normalization_strategy}"
            )
        return c_driver, c_response

    def transform_params(self, param_transform: Callable) -> bool:
        if self.perturb_stats:
            self.driver_stats = {
                k: param_transform(f"driver_{k}", v)
                for k, v in self.driver_stats.items()
            }
            self.response_stats = {
                k: param_transform(f"response_{k}", v)
                for k, v in self.response_stats.items()
            }
            self.c_driver, self.c_response = self._compute_normalization_constants()

        if self.random_seed is not None and self.randomize_driver_indices:
            self.driver_indices = self.rng.choice(
                max(self.driver_dim, self.response_dim),
                self.response_dim,
                replace=False,
            )

        return True

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        padded_driver = np.pad(driver, (0, max(self.response_dim - self.driver_dim, 0)))
        return (
<<<<<<< HEAD
            self.c_driver * padded_driver[self.driver_indices]
            + self.c_response * response
=======
            self.driver_scale * padded_driver[self.driver_indices]
            + self.response_scale * response
>>>>>>> 4e700808a5de9e440f87706e2aa01897a41c1d67
        )

    def jac(
        self, driver: np.ndarray, response: np.ndarray, wrt: str = "driver"
    ) -> np.ndarray:
        if wrt == "driver":
            djac = np.pad(
                np.eye(self.driver_dim),
                ((0, max(self.response_dim - self.driver_dim, 0)), (0, 0)),
            )
<<<<<<< HEAD
            return self.c_driver * djac[self.driver_indices]
        elif wrt == "response":
            rjac = np.eye(self.response_dim)
            return self.c_response * rjac
=======
            return self.driver_scale * djac[self.driver_indices]
        elif wrt == "response":
            rjac = np.eye(self.response_dim)
            return self.response_scale * rjac
>>>>>>> 4e700808a5de9e440f87706e2aa01897a41c1d67
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
