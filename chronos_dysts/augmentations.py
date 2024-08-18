"""Augmentations for dynamical systems

TODO: maybe add augmentations at different scales
- coordinate scale (should prob go into preprocess_entry)
- system scale
- ensemble of systems scale
"""
import numpy as np

from itertools import combinations
from gluonts.dataset import Dataset
from gluonts.dataset.common import ListDataset
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence, Tuple, Iterator


def stack_and_extract_metadata(dataset: Dataset) -> Tuple[np.ndarray, Tuple[Any]]:
    """Utility for unpacking gluonts dataset into array and extracting metadata
    """
    coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
    coordinates = np.stack(coords)
    return coordinates, metadata


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

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __iter__(self) -> Iterator:
        coordinates, metadata = stack_and_extract_metadata(self.dataset)
        coeffs = self.rng.dirichlet(self.alpha*np.ones(coordinates.shape[0]), size=self.num_combinations)
        return ({"start": metadata[0], "target": combo} for combo in coeffs@coordinates)


@dataclass
class RandomAffineTransform(Dataset):
    pass
    """Random affine transformations of coordinates with coefficients sampled from a zero-mean Gaussian

    NOTE: this augmentation is on the system scale (across coordinates)

    :param out_dim: output dimension of the linear map
    :param scale: gaussian distribution scale
    :param random_seed: RNG seed
    """
    dataset: Dataset
    out_dim: int
    scale: float
    random_seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __iter__(self) -> Iterator:
        coordinates, metadata = stack_and_extract_metadata(self.dataset)
        affine_transform = self.rng.normal(scale=self.scale, size=(self.out_dim, 1+coordinates.shape[0]))
        combos = affine_transform[:, :-1]@coordinates + affine_transform[:, -1, np.newaxis]
        return ({"start": metadata[0], "target": combination} for combination in combos)
        

def sample_index_pairs(
    size: int, num_pairs: int, rng: Optional[np.random.Generator] = None
) -> Iterator:
    """Sample pairs from an arbitrary sequence
    
    TODO: this is a pretty general util, move to utils
    TODO: add option to filter by dyst_name for sampled pairs?
    """
    num_total_pairs = size*(size-1)//2
    assert num_pairs <= num_total_pairs, (
        "Cannot sample more pairs than unique pairs."
    )
    sampled_pairs = (rng or np.random).choice(num_total_pairs, size=num_pairs, replace=False)
    all_pairs = list(combinations(range(size), 2))
    return (all_pairs[i] for i in sampled_pairs) 

     
@dataclass
class RandomProjectedSkewTransform:
    """
    Randomly combines pairs of timeseries and linearly maps them into common embedding space 
    Linear maps are zero mean gaussian random matrices

    NOTE: this is an example of an ensemble-scale transformation
    """
    dataset1: Dataset
    dataset2: Dataset
    embedding_dim: int  # embedding dimension for the skew projection
    scale: float  # scale for the gaussian random projection matrices
    random_seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __iter__(self) -> Iterator:
        coords1, metadata = stack_and_extract_metadata(self.dataset1)
        coords2, _ = stack_and_extract_metadata(self.dataset2)
        proj1 = self.rng.normal(scale=self.scale, size=(self.embedding_dim, coords1.shape[0]))
        proj2 = self.rng.normal(scale=self.scale, size=(self.embedding_dim, coords2.shape[0]))
        transformed = proj1@coords1 + proj2@coords2
        return ({"start": metadata[0], "target": coord} for coord in transformed)

        



