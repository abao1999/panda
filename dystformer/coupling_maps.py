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

    def _pad_and_index_driver(
        self, driver: np.ndarray, pad_jac: bool = False
    ) -> np.ndarray:
        """Pad and index driver to match the response dimension"""
        pad_spec = (0, max(self.response_dim - self.driver_dim, 0))
        if pad_jac:
            pad_spec = (pad_spec, (0, 0))
        return np.pad(driver, pad_spec)[self.driver_indices]

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

            djac = self._pad_and_index_driver(np.eye(self.driver_dim), pad_jac=True)
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
        TODO
        """
        inds = np.arange(self.response_dim)
        driver_ub_inds = self._permuted_driver_unbounded_indices(
            driver_unbounded_indices
        )

        # get mask of indices that dont intersect with any unbounded indices
        both_ub_inds = np.union1d(driver_ub_inds, response_unbounded_indices)
        both_ub_mask = ~np.isin(inds, both_ub_inds)

        # get mask of unbounded indices xored with
        driver_ub_mask = np.isin(inds, driver_ub_inds) ^ both_ub_mask
        response_ub_mask = np.isin(inds, response_unbounded_indices) ^ both_ub_mask

        # this is pure indexing magic
        # given the response, reorganizes the coords back into the driver space
        # then applies postprocessing in the driver space
        # then undoes the organization back into the response space
        # Works for driver_dim > response_dim and driver_dim < response_dim cases
        driver_coords = np.zeros(self.response_dim)
        if driver_postprocess_fn is not None:
            driver_coords = np.zeros(self.driver_dim)
            sort_perm = np.argsort(self.driver_indices)
            sorted_inds = self.driver_indices[sort_perm]
            driver_coords[sorted_inds[: self.driver_dim]] = response[sort_perm][
                : self.driver_dim
            ]
            driver_coords = driver_postprocess_fn(driver_coords)
            driver_coords = self._pad_and_index_driver(driver_coords)

        response_coords = np.zeros(self.response_dim)
        if response_postprocess_fn is not None:
            response_coords = response_postprocess_fn(response)

        # combine the masked driver and response with an average
        # if two indices are both unbounded, the average of the two is returned
        # if only one is unbounded, the unbounded index is returned (via the mask)
        # if neither is unbounded, theyre the same and the average is either one of them
        return 0.5 * (
            driver_coords * driver_ub_mask + response_coords * response_ub_mask
        )


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
