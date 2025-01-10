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

    def _serialize(self) -> dict:
        """Serialize (JSON compatible) coupling map data for saving params"""
        raise NotImplementedError

    @staticmethod
    def _deserialize(data: dict) -> None:
        """Deserialize (JSON compatible) coupling map data for setting params"""
        raise NotImplementedError


@dataclass
class RandomAdditiveCouplingMap(BaseCouplingMap):
    """
    Simple additive coupling map between driver and response flows
    Optionally, randomly samples the indices of the driver and response flows to couple.

    NOTE: the dimensions are fixed to the response system
    """

    driver_scale: float | np.ndarray = 1.0
    response_scale: float | np.ndarray = 1.0

    transform_scales: bool = True
    randomize_driver_indices: bool = True

    random_seed: int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.driver_scale, (float, np.ndarray))
        assert isinstance(self.response_scale, (float, np.ndarray))

        if isinstance(self.driver_scale, np.ndarray):
            assert self.driver_scale.shape == (self.driver_dim,)
        if isinstance(self.response_scale, np.ndarray):
            assert self.response_scale.shape == (self.response_dim,)

        if self.random_seed is not None:
            self.rng = np.random.default_rng(self.random_seed)

        self.driver_indices = np.arange(self.response_dim)

    @property
    def n_params(self) -> int:
        # 2 params for the scales + self.response_dim params for the indices
        return 2 * (self.transform_scales) + (self.response_dim) * (
            self.random_seed is not None and self.randomize_driver_indices
        )

    def _permuted_driver_unbounded_indices(
        self, driver_indices_ub: list[int]
    ) -> list[int]:
        """Gets the unbounded indices relative to the permutation of self.driver_indices"""
        return [
            i for i, ind in enumerate(self.driver_indices) if ind in driver_indices_ub
        ]

    def unbounded_indices(
        self, driver_indices_ub: list[int], response_indices_ub: list[int]
    ) -> list[int]:
        """
        Return indices of the coupling map shifted by the driver dimension

        Indices are derived from overlapping driver and response unbounded indices
        """
        # get the subset of indices of self.driver_indices which have elements in driver_indices_ub
        driver_inds_subset = self._permuted_driver_unbounded_indices(driver_indices_ub)

        # it's impossible to tell which indices are unbounded without integrating the system
        # so we loosely assume that the unbounded indices are the union of
        #  1. the indices of self.driver_indices which have elements in driver_indices_ub
        #  2. the response unbounded indices
        # note that the post processing method can optionally selectively bound only a subset
        # of these proposed unbounded indices, and need not adhere strictly to these
        return list(set(driver_inds_subset) | set(response_indices_ub))

    def transform_params(self, param_transform: Callable) -> bool:
        if self.transform_scales:
            if isinstance(self.driver_scale, float) and isinstance(
                self.response_scale, float
            ):
                self.driver_scale = param_transform("driver_scale", self.driver_scale)
                self.response_scale = param_transform(
                    "response_scale", self.response_scale
                )
            elif isinstance(self.driver_scale, np.ndarray) and isinstance(
                self.response_scale, np.ndarray
            ):
                self.driver_scale = np.asarray(
                    [
                        param_transform(f"driver_scale_{p}", self.driver_scale[p])
                        for p in range(self.driver_dim)
                    ]
                )
                self.response_scale = np.asarray(
                    [
                        param_transform(f"response_scale_{p}", self.response_scale[p])
                        for p in range(self.response_dim)
                    ]
                )

        if self.random_seed is not None and self.randomize_driver_indices:
            self.driver_indices = self.rng.choice(
                max(self.driver_dim, self.response_dim),
                self.response_dim,
                replace=False,
            )

        return True

    def _pad_and_index_driver(self, driver: np.ndarray, axis: int = 0) -> np.ndarray:
        """Pad (append) and index driver to match the response dimension"""
        assert (
            axis < driver.ndim
        ), f"axis {axis} must be less than driver.ndim {driver.ndim}"
        pad_spec = (0, max(self.response_dim - self.driver_dim, 0))
        pad_spec = ((0, 0),) * axis + (pad_spec,) + ((0, 0),) * (driver.ndim - axis - 1)
        padded_driver = np.pad(driver, pad_spec)
        return np.take(padded_driver, self.driver_indices, axis=axis)

    def __call__(self, driver: np.ndarray, response: np.ndarray) -> np.ndarray:
        padded_driver = self._pad_and_index_driver(driver)

        if isinstance(self.driver_scale, np.ndarray):
            # reuse padding and indexing from driver
            driver_scale = self._pad_and_index_driver(self.driver_scale)
        else:
            driver_scale = self.driver_scale

        return driver_scale * padded_driver + self.response_scale * response

    def jac(
        self, driver: np.ndarray, response: np.ndarray, wrt: str = "driver"
    ) -> np.ndarray:
        if wrt == "driver":
            if isinstance(self.driver_scale, np.ndarray):
                # reuse padding and indexing from driver
                driver_scale = self._pad_and_index_driver(self.driver_scale)
                driver_scale = driver_scale[:, np.newaxis]
            else:
                driver_scale = self.driver_scale

            djac = self._pad_and_index_driver(np.eye(self.driver_dim))
            return driver_scale * djac
        elif wrt == "response":
            rjac = np.eye(self.response_dim)
            return self.response_scale * rjac
        else:
            raise ValueError(f"Invalid wrt argument: {wrt}")

    def _postprocessing(
        self,
        response: np.ndarray,
        driver_postprocess_fn: Callable | None = None,
        response_postprocess_fn: Callable | None = None,
        response_unbounded_indices: list[int] = [],
        driver_unbounded_indices: list[int] = [],
    ) -> np.ndarray:
        """
        Dynamical systems can have unbounded coordinates e.g. time-like drivers or exponential growth dimensions
        Postprocessing is used to bound these coordinates, for instance by making them periodic
        For skew systems, this postprocessing is complicated by two considerations:
            a. Our feature to allow randomized shuffling of driver dimensions
            b. The driver and response systems live in different spaces.
        Therefore, a roadmap of the postprocessing is as follows:
            1. Keep track of the unbounded indices for the driver and response systems
            2. Map the driver system into the driver space and apply the driver's postprocessing
            3. Apply the response's postprocessing to the response system
            4. Map the postprocessed driver back into the response space
            5. Enforce consistency between the separately postprocessed driver and response
                This is done by a heuristic scheme that aggregates the driver and response postprocessed coordinates:
                    + If two indices are both unbounded, the average of the two is returned
                    + If only one is unbounded, the unbounded index is returned (via the mask)
                    + If neither is unbounded, theyre the same and the average is either one of them
        """
        inds = np.arange(self.response_dim)
        driver_ub_inds = self._permuted_driver_unbounded_indices(
            driver_unbounded_indices
        )

        # get mask of indices that dont intersect with any unbounded indices
        both_ub_inds = np.union1d(driver_ub_inds, response_unbounded_indices)
        both_ub_mask = ~np.isin(inds, both_ub_inds)

        # get mask of unbounded indices xored with the both_ub_mask
        driver_ub_mask = np.isin(inds, driver_ub_inds) ^ both_ub_mask
        response_ub_mask = np.isin(inds, response_unbounded_indices) ^ both_ub_mask

        # this is pure indexing magic
        # given the response, reorganizes the coords back into the driver space
        # then applies postprocessing in the driver space
        # then undoes the organization back into the response space
        driver = response.copy()
        if driver_postprocess_fn is not None:
            driver = np.zeros((self.driver_dim, *response.shape[1:]))
            sort_perm = np.argsort(self.driver_indices)
            sorted_inds = self.driver_indices[sort_perm]
            driver[sorted_inds[: self.driver_dim]] = response[sort_perm][
                : self.driver_dim
            ]
            driver = np.asarray(driver_postprocess_fn(*driver))
            driver = self._pad_and_index_driver(driver)

        if response_postprocess_fn is not None:
            response = np.asarray(response_postprocess_fn(*response))

        # combine the masked driver and response with an average
        # if two indices are both unbounded, the average of the two is returned
        # if only one is unbounded, the unbounded index is returned (via the mask)
        # if neither is unbounded, theyre the same and the average is performs the identity
        driver_mask = driver_ub_mask[..., np.newaxis, np.newaxis]
        response_mask = response_ub_mask[..., np.newaxis, np.newaxis]
        return 0.5 * (driver * driver_mask + response * response_mask)

    def _serialize(self) -> dict:
        """Serialize (JSON compatible) coupling map parameters for saving params"""
        driver_scale = (
            self.driver_scale.tolist()
            if isinstance(self.driver_scale, np.ndarray)
            else self.driver_scale
        )
        response_scale = (
            self.response_scale.tolist()
            if isinstance(self.response_scale, np.ndarray)
            else self.response_scale
        )
        return {
            "preinit": {
                "driver_dim": self.driver_dim,
                "response_dim": self.response_dim,
                "driver_scale": driver_scale,
                "response_scale": response_scale,
                "random_seed": self.random_seed,
                "randomize_driver_indices": self.randomize_driver_indices,
                "transform_scales": self.transform_scales,
            },
            "postinit": {
                "driver_indices": self.driver_indices.tolist(),
            },
        }

    @classmethod
    def _deserialize(cls, data: dict) -> "RandomAdditiveCouplingMap":
        """Deserialize (JSON compatible) coupling map parameters for reconstructing coupling map"""
        preinit_data = data["preinit"]
        for key in ["driver_scale", "response_scale"]:
            if isinstance(preinit_data[key], list):
                preinit_data[key] = np.array(preinit_data[key])

        obj = cls(**preinit_data)
        obj.driver_indices = np.array(data["postinit"]["driver_indices"])
        return obj


@dataclass
class RandomLinearCouplingMap(BaseCouplingMap):
    """
    Affine coupling map between driver and response flows

    TODO: this is very outdated
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
