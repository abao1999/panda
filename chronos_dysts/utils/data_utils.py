import numpy as np

from itertools import combinations
from gluonts.dataset import Dataset
from typing import Optional, Any, Tuple, Iterator


def stack_and_extract_metadata(dataset: Dataset) -> Tuple[np.ndarray, Tuple[Any]]:
    """Utility for unpacking gluonts dataset into array and extracting metadata
    """
    coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
    coordinates = np.stack(coords)
    return coordinates, metadata

def sample_index_pairs(
    size: int, num_pairs: int, rng: Optional[np.random.Generator] = None
) -> Iterator:
    """Sample pairs from an arbitrary sequence
    TODO: add option to filter by dyst_name for sampled pairs?
    """
    num_total_pairs = size*(size-1)//2
    assert num_pairs <= num_total_pairs, (
        "Cannot sample more pairs than unique pairs."
    )
    sampled_pairs = (rng or np.random).choice(num_total_pairs, size=num_pairs, replace=False)
    all_pairs = list(combinations(range(size), 2))
    return (all_pairs[i] for i in sampled_pairs) 

# stationarity tests