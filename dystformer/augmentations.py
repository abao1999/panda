"""
Training augmentations for multivariate time series
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import Array


@dataclass
class RandomConvexCombinationTransform(Dataset):
    """Random convex combinations of coordinates with coefficients sampled from a dirichlet distribution

    NOTE: this augmentation is on the system scale (across coordinates)

    :param num_combinations: number of random convex combinations to sample
    :param alpha: dirichlet distribution scale
    :param random_seed: RNG seed
    """

    dataset: Dataset
    num_combinations: int
    alpha: float
    random_seed: Optional[int] = 0
    split_coords: bool = False

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: Array) -> Array:
        coeffs = self.rng.dirichlet(
            self.alpha * np.ones(coordinates.shape[0]), size=self.num_combinations
        )
        return coeffs @ timeseries


@dataclass
class RandomAffineTransform(Dataset):
    pass
    """Random affine transformations of coordinates with coefficients sampled from a zero-mean Gaussian

    :param out_dim: output dimension of the linear map
    :param scale: gaussian distribution scale
    :param random_seed: RNG seed
    """
    out_dim: int
    scale: float
    random_seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: Array) -> Array:
        affine_transform = self.rng.normal(
            scale=self.scale, size=(self.out_dim, 1 + timeseries.shape[0])
        )
        return affine_transform[:, :-1] @ timeseries + affine_transform[:, -1, np.newaxis]


@dataclass
class RandomProjectedSkewTransform:
    """
    Randomly combines pairs of timeseries and linearly maps them into a common embedding space
    Linear maps are zero mean gaussian random matrices

    :param embedding_dim: embedding dimension for the skew projection
    :param scale: scale for the gaussian random projection matrices
    :param random_seed: RNG seed

    TODO: figure out how to make this on-the-fly
    """
    embedding_dim: int
    scale: float
    random_seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries1: Array, timeseries2: Array) -> Array:
        proj1 = self.rng.normal(
            scale=self.scale, size=(self.embedding_dim, timeseries1.shape[0])
        )
        proj2 = self.rng.normal(
            scale=self.scale, size=(self.embedding_dim, timeseries2.shape[0])
        )
        return proj1 @ timeseries1 + proj2 @ timeseries2
