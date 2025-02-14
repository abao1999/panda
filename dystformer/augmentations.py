"""
Training augmentations for multivariate time series
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RandomDimSelectionTransform:
    """Randomly select a subset of dimensions from a timeseries"""

    num_dims: int
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray, axis: int = 0) -> NDArray:
        selected_dims = self.rng.choice(
            timeseries.shape[axis], self.num_dims, replace=False
        )
        return np.take(timeseries, selected_dims, axis=axis)


@dataclass
class FixedDimensionDelayEmbeddingTransform:
    """Delays embeds a timeseries to a fixed embedding dimension

    NOTE:
        - if the embedding dimension is non permissible (e.g. more delays than timepoints), an error is raised
        - if the dimension of the timeseries is greater than the embedding dimension, then embedding dimensions
          are randomly sampled from the channel dimension

    :param embedding_dim: embedding dimension for the delay embeddings
    :param channel_dim: dimension corresponding to channels
    :param time_dim: dimension corresponding to timepoints
    """

    embedding_dim: int
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        """
        :param timeseries: (num_channels, num_timepoints) timeseries to delay embed
        """
        assert timeseries.shape[1] > self.embedding_dim, (
            "Embedding dimension cannot be larger than the number of timepoints"
        )
        num_channels = timeseries.shape[0]
        per_dim_embed_dim = (self.embedding_dim - num_channels) // num_channels
        remaining_dims = (self.embedding_dim - num_channels) % num_channels

        if num_channels >= self.embedding_dim:
            selected_dims = self.rng.choice(
                num_channels, self.embedding_dim, replace=False
            )
            return timeseries[selected_dims]

        # Distribute remaining dimensions evenly
        extra_dims = [1 if i < remaining_dims else 0 for i in range(num_channels)]

        delay_embeddings = [
            np.stack(
                [
                    np.roll(timeseries[i], shift)
                    for shift in range(1, 1 + per_dim_embed_dim + extra)
                ]
            )[:, 1 + per_dim_embed_dim :]
            for i, extra in enumerate(extra_dims)
            if extra > 0 or per_dim_embed_dim > 0
        ]

        return np.vstack([timeseries[:, 1 + per_dim_embed_dim :], *delay_embeddings])


@dataclass
class QuadraticEmbeddingTransform:
    """Embed the channel dimension with a quadratic basis

    NOTE: does the same thing as PatchTSTPolynomialEmbedding with degree=2,
    but much more efficiently
    """

    def __call__(self, timeseries: NDArray) -> NDArray:
        """
        Parameters:
            timeseries (`NDArray` of shape `(num_channels, num_timepoints)`, *required*):
                Timeseries to embed
        """
        indices = np.triu_indices(timeseries.shape[0])
        features = timeseries[indices[0]] * timeseries[indices[1]]
        return np.vstack([timeseries, features])


@dataclass
class RandomConvexCombinationTransform:
    """Random convex combinations of coordinates with coefficients sampled from a dirichlet distribution

    :param num_combinations: number of random convex combinations to sample
    :param alpha: dirichlet distribution scale
    :param random_seed: RNG seed
    """

    num_combinations: int
    alpha: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        coeffs = self.rng.dirichlet(
            self.alpha * np.ones(timeseries.shape[0]), size=self.num_combinations
        )
        return coeffs @ timeseries


@dataclass
class RandomAffineTransform:
    """Random affine transformations of coordinates with coefficients sampled from a zero-mean Gaussian

    :param out_dim: output dimension of the linear map
    :param scale: gaussian distribution scale
    :param random_seed: RNG seed
    """

    out_dim: int
    scale: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        affine_transform = self.rng.normal(
            scale=self.scale, size=(self.out_dim, 1 + timeseries.shape[0])
        )
        return (
            affine_transform[:, :-1] @ timeseries + affine_transform[:, -1, np.newaxis]
        )


@dataclass
class RandomProjectedSkewTransform:
    """
    Randomly combines pairs of timeseries and linearly maps them into a common embedding space
    Linear maps are zero mean gaussian random matrices

    :param embedding_dim: embedding dimension for the skew projection
    :param scale: scale for the gaussian random projection matrices
    :param random_seed: RNG seed

    TODO:
        - figure out how to make this on-the-fly
        - maybe even deprecate this, is chaoticity preserved?
    """

    embedding_dim: int
    scale: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries1: NDArray, timeseries2: NDArray) -> NDArray:
        proj1 = self.rng.normal(
            scale=self.scale, size=(self.embedding_dim, timeseries1.shape[0])
        )
        proj2 = self.rng.normal(
            scale=self.scale, size=(self.embedding_dim, timeseries2.shape[0])
        )
        return proj1 @ timeseries1 + proj2 @ timeseries2
