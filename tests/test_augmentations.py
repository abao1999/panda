import argparse
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from gluonts.dataset import Dataset
from gluonts.dataset.common import FileDataset

from dystformer import augmentations
from dystformer.utils import (
    get_system_filepaths,
    plot_trajs_multivariate,
    sample_index_pairs,
    stack_and_extract_metadata,
)

AUG_CLS_DICT = {
    "system_scale": ["RandomConvexCombinationTransform", "RandomAffineTransform"],
    "ensemble_scale": ["RandomProjectedSkewTransform"],
}
AUG_CLS_KWARGS = {
    "RandomConvexCombinationTransform": {
        "num_combinations": 10,
        "alpha": 0.6,
        "random_seed": 0,
    },
    "RandomAffineTransform": {"out_dim": 6, "scale": 5e-1, "random_seed": 0},
    "RandomProjectedSkewTransform": {
        "embedding_dim": 6,
        "scale": 5e-1,
        "random_seed": 0,
    },
}

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def get_dyst_datasets(
    dyst_name: str, split: str, one_dim_target: bool = True
) -> List[Dataset]:
    """
    Returns list of datasets associated with dyst_name, converted to FileDataset
    """
    filepaths = get_system_filepaths(dyst_name, base_dir=DATA_DIR, split=split)

    gts_datasets_list = []
    for filepath in filepaths:
        gts_dataset = FileDataset(
            path=Path(filepath), freq="h", one_dim_target=one_dim_target
        )
        gts_datasets_list.append(gts_dataset)
    return gts_datasets_list


def get_dysts_datasets_dict(
    dysts_names: List[str], split: str, one_dim_target: bool = True
) -> Dict[str, List[Dataset]]:
    """
    Returns a dictionary with key as dyst_name and value as list of FileDatasets loaded from that dyst_name folder
    """
    gts_datasets_dict = defaultdict(list)
    for dyst_name in dysts_names:
        gts_datasets_dict[dyst_name] = get_dyst_datasets(
            dyst_name, split=split, one_dim_target=one_dim_target
        )
    assert list(gts_datasets_dict.keys()) == dysts_names, "Mismatch in dyst names"
    return gts_datasets_dict


def accumulate_dyst_samples(
    dyst_name: str,
    gts_datasets_dict: Dict[str, List[Dataset]],
    augmentation_fn: Optional[Callable[[Dataset], Dataset]] = None,
) -> np.ndarray:
    """
    Accumulate samples from all datasets associated with dyst_name
    Params:
        augmentation_fn: System-scale augmentation function that takes in GluonTS Dataset and returns Iterator
    Returns a numpy array of shape (num_samples, num_dims, num_timesteps)
    """
    dyst_coords_samples = []
    # loop through all sample files for dyst_name system
    for gts_dataset in gts_datasets_dict[dyst_name]:
        if augmentation_fn is not None:
            # Apply augmentation, which takes in GluonTS Dataset and returns ListDataset
            gts_dataset = augmentation_fn(gts_dataset)

        # extract the coordinates
        dyst_coords, _ = stack_and_extract_metadata(
            gts_dataset,
        )
        dyst_coords_samples.append(dyst_coords)
        print("data shape: ", dyst_coords.shape)

    dyst_coords_samples = np.array(dyst_coords_samples)
    print(dyst_coords_samples.shape)
    return dyst_coords_samples


def apply_augmentations_system(
    dysts_names: List[str], split: str = "train", one_dim_target: bool = False
) -> None:
    """
    Apply augmentations on the system scale
    """
    gts_datasets_dict = get_dysts_datasets_dict(
        dysts_names, split=split, one_dim_target=one_dim_target
    )
    for dyst_name in dysts_names:
        # for every system-scale augmentation
        for augmentation_cls_name in AUG_CLS_DICT["system_scale"]:
            print(augmentation_cls_name)
            augmentation_cls = getattr(augmentations, augmentation_cls_name)
            print("Applying system-scale augmentation: ", augmentation_cls.__name__)
            kwargs = AUG_CLS_KWARGS[augmentation_cls_name]
            print("kwargs: ", kwargs)

            # build augmentation partial function
            augmentation_fn = partial(
                augmentation_cls, split_coords=one_dim_target, **kwargs
            )
            # accumulate coords in sample dimension, while applying augmentation to each coords
            dyst_coords_samples = accumulate_dyst_samples(
                dyst_name,
                gts_datasets_dict,
                augmentation_fn,
            )

            plot_trajs_multivariate(
                dyst_coords_samples,
                save_dir="tests/figs",
                plot_name=f"{dyst_name}_{augmentation_cls_name}",
            )


def apply_augmentations_ensemble(
    dysts_names: List[str],
    split: str = "train",
    num_pairs_dysts: int = 1,
    one_dim_target: bool = False,
) -> None:
    """
    Apply augmentations on the ensemble scale, given dict that maps dyst_name to list of all associated datasets
    Restriction to only combining pairs of dysts, and only along sample dimension
        i.e. dyst1 sample i combined with dyst2 sample i
                where sample i is assumed to be consistent (e.g. same initial conditions, parameter perturbs) across all dysts
    """
    gts_datasets_dict = get_dysts_datasets_dict(
        dysts_names, split=split, one_dim_target=one_dim_target
    )
    # for every ensemble-scale augmentation
    for augmentation_cls_name in AUG_CLS_DICT["ensemble_scale"]:
        print(augmentation_cls_name)
        augmentation_cls = getattr(augmentations, augmentation_cls_name)
        print("Applying ensemble-scale augmentation: ", augmentation_cls.__name__)
        kwargs = AUG_CLS_KWARGS[augmentation_cls_name]
        print("kwargs: ", kwargs)

        # build augmentation partial function
        augmentation_fn = partial(
            augmentation_cls, split_coords=one_dim_target, **kwargs
        )

        # for every pair of dysts
        for i, j in sample_index_pairs(len(dysts_names), num_pairs=num_pairs_dysts):
            # TODO: wrap this in a helper function
            dyst1, dyst2 = dysts_names[i], dysts_names[j]
            print(f"Applying ensemble-scale augmentation to {dyst1} and {dyst2}")
            num_samples = min(
                len(gts_datasets_dict[dyst1]), len(gts_datasets_dict[dyst2])
            )

            dyst_pair_coords_samples = []
            for sample_idx in range(num_samples):
                print(f"Augmenting sample index {sample_idx}")
                gts_dataset = augmentation_fn(
                    gts_datasets_dict[dyst1][sample_idx],
                    gts_datasets_dict[dyst2][sample_idx],
                )

                dyst_pair_coords, _ = stack_and_extract_metadata(
                    gts_dataset,
                )
                dyst_pair_coords_samples.append(dyst_pair_coords)
                print("augmented data shape: ", dyst_pair_coords.shape)

            dyst_pair_coords_samples = np.array(dyst_pair_coords_samples)
            print(dyst_pair_coords_samples.shape)

            plot_trajs_multivariate(
                dyst_pair_coords_samples,
                save_dir="tests/figs",
                plot_name=f"{'_'.join([dyst1, dyst2])}_{augmentation_cls_name}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dysts_names", help="Names of the dynamical systems", nargs="+", type=str
    )
    parser.add_argument("--split", help="Split of the data", type=str, default=None)
    parser.add_argument(
        "--one_dim_target", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    dysts_names = args.dysts_names

    print("Applying system-scale transformations")
    apply_augmentations_system(
        dysts_names, split=args.split, one_dim_target=args.one_dim_target
    )

    print("Applying ensemble-scale transformations")
    apply_augmentations_ensemble(
        dysts_names,
        split=args.split,
        num_pairs_dysts=1,
        one_dim_target=args.one_dim_target,
    )
