"""Augmentations for dynamical systems

TODO: maybe add augmentations at different scales
- coordinate scale (should prob go into preprocess_entry)
- system scale
- ensemble of systems scale
"""
import numpy as np

from itertools import combinations
from abc import abstractmethod
from gluonts.dataset import Dataset
from gluonts.dataset.common import ListDataset
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence, Tuple


def stack_and_extract_metadata(dataset: Dataset) -> Tuple[np.ndarray, Tuple[Any]]:
    """Utility for unpacking gluonts dataset into array and extracting metadata
    """
    coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
    coordinates = np.stack(coords)
    return coordinates, metadata

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
        coordinates, metadata = stack_and_extract_metadata(dataset)
        coeffs = self.rng.dirichlet(self.alpha*np.ones(coordinates.shape[0]), size=self.num_combinations)
        combos = coeffs@coordinates
        data = ({"start": metadata[0], "target": combination} for combination in combos)
        return ListDataset(data, freq='h')
        

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
        coordinates, metadata = stack_and_extract_metadata(dataset)
        affine_transform = self.rng.normal(scale=self.scale, size=(self.out_dim, 1+coordinates.shape[0]))
        combos = affine_transform[:, :-1]@coordinates + affine_transform[:, -1, np.newaxis]
        data = ({"start": metadata[0], "target": combination} for combination in combos)
        return ListDataset(data, freq='h')
        

@dataclass
class RandomProjectedSkewTransform:
    """
    Randomly combines pairs of timeseries and linearly maps them into common embedding space 
    Linear maps are zero mean gaussian random matrices

    NOTE: this is an example of an ensemble-scale transformation
    """
    num_skew_pairs: int  # number of pairs of systems to sample
    embedding_dim: int  # embedding dimension for the skew projection
    scale: float  # scale for the gaussian random projection matrices
    random_seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def random_project(self, sys1: Dataset, sys2: Dataset) -> List[Dict[str, Any]]:
        coords1, metadata = stack_and_extract_metadata(sys1)
        coords2, _ = stack_and_extract_metadata(sys2)
        proj1 = self.rng.normal(scale=self.scale, size=(self.embedding_dim, coords1.shape[0]))
        proj2 = self.rng.normal(scale=self.scale, size=(self.embedding_dim, coords2.shape[0]))
        transformed = proj1@coords1 + proj2@coords2
        data = ({"start": metadata[0], "target": coord} for coord in transformed)
        return ListDataset(data, freq='h')

    def __call__(self, datasets: Sequence[Dataset]) -> List[ListDataset]:
        num_total_pairs = len(datasets)*(len(datasets)-1)//2
        assert self.num_skew_pairs < num_total_pairs, (
            "Cannot sample more skew pairs than unique pairs."
        )

        pair_inds = self.rng.choice(num_total_pairs, size=self.num_skew_pairs, replace=False)
        all_pairs = list(combinations(range(len(datasets)), 2))
        sampled_pairs = [all_pairs[i] for i in pair_inds]
        return [self.random_project(datasets[i], datasets[j]) for i,j in sampled_pairs]

        



