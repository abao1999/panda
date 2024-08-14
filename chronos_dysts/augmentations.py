"""Augmentations for dynamical systems

TODO: maybe add augmentations at different scales
- coordinate scale (should prob go into preprocess_entry)
- system scale
- ensemble of systems scale
"""
import numpy as np

from abc import abstractmethod
from gluonts.dataset import Dataset
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


class SystemTransform:
    """Abstract class for transformations on the system scale (across coordinates in a single system)
    """

    @abstractmethod
    def __call__(self, dataset: Dataset) -> List[Dict[str, Any]]:
        raise NotImplementedError


class IdentityTransform(SystemTransform):
    """Dummy do-nothing system transform. Kinda dumb might delete idk
    """
    def __call__(self, dataset: Dataset) -> List[Dict[str, Any]]:
        return dataset


@dataclass
class RandomConvexCombinationTransform(SystemTransform):
    """Random convex combinations of coordinates with coefficients sampled from a dirichlet distribution

    NOTE: this augmentation is on the system scale (across coordinates)

    :param num_combinations: number of random convex combinations to sample
    :param alpha: dirichlet distribution scale
    :param random_seed: RNG seed
    """
    num_combinations: int
    alpha: float
    random_seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, dataset: Dataset) -> List[Dict[str, Any]]:
        coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
        coordinates = np.stack(coords)
        coeffs = self.rng.dirichlet(self.alpha*np.ones(coordinates.shape[0]), size=self.num_combinations)
        combos = coeffs@coordinates
        return [{"start": metadata[0], "target": combination} for combination in combos]
        

@dataclass
class RandomAffineTransform(SystemTransform):
    """Random affine transformations of coordinates with coefficients sampled from a zero-mean Gaussian

    NOTE: this augmentation is on the system scale (across coordinates)

    :param out_dim: output dimension of the linear map
    :param scale: gaussian distribution scale
    :param random_seed: RNG seed
    """
    out_dim: int
    scale: float
    random_seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, dataset: Dataset) -> List[Dict[str, Any]]:
        coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
        coordinates = np.stack(coords)
        affine_transform = self.rng.normal(scale=self.scale, size=(self.out_dim, 1+coordinates.shape[0]))
        combos = affine_transform[:, :-1]@coordinates + affine_transform[:, -1, np.newaxis]
        return [{"start": metadata[0], "target": combination} for combination in combos]
        
