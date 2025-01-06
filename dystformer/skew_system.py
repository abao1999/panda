"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import logging
from typing import Callable

import numpy as np
from dysts.base import DynSys

from dystformer.coupling_maps import RandomAdditiveCouplingMap

logger = logging.getLogger(__name__)


class SkewProduct(DynSys):
    def __init__(
        self,
        driver: DynSys,
        response: DynSys,
        coupling_map: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        _default_random_seed: int | None = None,
        _postprocess_response_only: bool = False,
        **kwargs,
    ):
        """A skew-product dynamical system composed of a driver and response system.

        The driver system evolves independently while the response system is influenced by the driver
        through a coupling map. Inherits from dysts.base.DynSys to enable trajectory generation.

        Args:
            driver (DynSys): The autonomous driver dynamical system
            response (DynSys): The response dynamical system that is influenced by the driver
            coupling_map (Callable, optional):
                Function defining how driver and response systems are coupled.
                If None, defaults to RandomAdditiveCouplingMap.
            _default_random_seed (int, optional):
                Random seed for default coupling map initialization
        """
        # default to additively forcing the response with the driver
        if coupling_map is None:
            self.coupling_map = RandomAdditiveCouplingMap(
                driver.dimension, response.dimension, random_seed=_default_random_seed
            )
        else:
            self.coupling_map = coupling_map

        super().__init__(
            metadata_path=None,
            parameters={},  # dummy: parameters are handled in the overwritten methods below
            dt=min(driver.dt, response.dt),
            period=max(driver.period, response.period),
            unbounded_indices=driver.unbounded_indices
            + [i + driver.dimension for i in response.unbounded_indices],
            metadata={
                "name": f"{driver.name}_{response.name}",
                "dimension": driver.dimension + response.dimension,
                "driver": driver,
                "response": response,
                "driver_dim": driver.dimension,
                "response_dim": response.dimension,
            },
            **kwargs,
        )

        self._postprocess_response_only = _postprocess_response_only

        # hack: set a dummy param list for param count checks
        n_params = len(driver.parameters) + len(response.parameters)
        if hasattr(self.coupling_map, "n_params"):
            n_params += self.coupling_map.n_params
        self.param_list = [0 for _ in range(n_params)]

        assert hasattr(driver, "ic") and hasattr(response, "ic"), (
            "Driver and response must have default initial conditions"
            "and must be of the same dimension"
        )

        self.ic = np.concatenate([self.driver.ic, self.response.ic])
        self.mean = np.concatenate([self.driver.mean, self.response.mean])
        self.std = np.concatenate([self.driver.std, self.response.std])

    def transform_params(self, param_transform: Callable):
        driver_success = self.driver.transform_params(param_transform)
        response_success = self.response.transform_params(param_transform)
        success = driver_success and response_success

        if hasattr(self.coupling_map, "transform_params"):
            success &= self.coupling_map.transform_params(param_transform)

        return success

    def has_jacobian(self):
        return self.driver.has_jacobian() and self.response.has_jacobian()

    def rhs(self, X: np.ndarray, t: float) -> np.ndarray:
        driver, response = X[: self.driver_dim], X[self.driver_dim :]
        driver_rhs = np.asarray(self.driver.rhs(driver, t))
        response_rhs = np.asarray(self.response.rhs(response, t))
        coupled_rhs = self.coupling_map(driver_rhs, response_rhs)
        return np.concatenate([driver_rhs, coupled_rhs])

    def jac(self, X: np.ndarray, t: float) -> np.ndarray:
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

    def __call__(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.rhs(X, t)

    def _postprocessing(self, *X: np.ndarray) -> np.ndarray:
        driver, response = X[: self.driver_dim], X[self.driver_dim :]
        if hasattr(self.driver, "_postprocessing"):
            driver = self.driver._postprocessing(*driver)
        if hasattr(self.response, "_postprocessing"):
            response = self.response._postprocessing(*response)
        if self._postprocess_response_only:
            return np.asarray(response)
        return np.concatenate([driver, response])
