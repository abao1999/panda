import numpy as np
import argparse
from pathlib import Path
from functools import partial
from typing import List

from gluonts.dataset.common import FileDataset, ListDataset

import importlib
from collections import defaultdict
from chronos_dysts.augmentations import stack_and_extract_metadata, sample_index_pairs
from chronos_dysts.utils import (
    get_dyst_filepaths, 
    plot_trajs_univariate, 
    plot_trajs_multivariate,
)


# list of augmentation classes to apply
augmentation_cls_dict = {
    "system_scale": ["RandomConvexCombinationTransform", "RandomAffineTransform"],
    "ensemble_scale": ["RandomProjectedSkewTransform"],
}

augmentation_cls_kwargs = {
    "RandomConvexCombinationTransform": {"num_combinations": 10, "alpha": 0.6, "random_seed": 0},
    "RandomAffineTransform": {"out_dim": 10, "scale": 1e-2, "random_seed": 0},
    "RandomProjectedSkewTransform": {"embedding_dim": 10, "scale": 1e-2, "random_seed": 0},
}

augmentations_module = importlib.import_module("chronos_dysts.augmentations")


def apply_augmentations_system(dysts_names: List[str]):
    """
    Apply augmentations on the system scale
    """
    print(dysts_names)
    num_systems = len(dysts_names)
    gts_datasets_dict = defaultdict(list)

    # for every dyst
    for i in range(num_systems):

        # load dyst data from corresponding filepath
        dyst_name = dysts_names[i]
        filepaths = get_dyst_filepaths(dyst_name)
        
        # for every file in the directory
        for filepath in filepaths:
            # create dataset by reading directly from filepath into FileDataset
            gts_dataset = FileDataset(path=Path(filepath), freq="h", one_dim_target=True) # TODO: consider other frequencies?
            # TODO: make this a dictionary, and also option to add augmented data to the dictionary for ensemble-scale transformation?
            gts_datasets_dict[dyst_name].append(gts_dataset) # save for ensemble-scale transformations

        # for every system-scale augmentation
        for augmentation_cls_name in augmentation_cls_dict["system_scale"]:
            print(augmentation_cls_name)
            augmentation_cls = getattr(augmentations_module, augmentation_cls_name)
            print("Applying system-scale augmentation: ", augmentation_cls.__name__)
            kwargs = augmentation_cls_kwargs[augmentation_cls_name]
            print("kwargs: ", kwargs)

            dyst_coords_samples = []
            # loop through all sample files for dyst_name system
            for gts_dataset in gts_datasets_dict[dyst_name]:
                # Apply augmentation, which takes in GluonTS Dataset and returns ListDataset
                gts_dataset_augmented = partial(
                    augmentation_cls,
                    **kwargs,
                )(gts_dataset)

                # extract the coordinates
                dyst_coords, _ = stack_and_extract_metadata(gts_dataset_augmented)
                dyst_coords_samples.append(dyst_coords)

                print("augmented data shape: ", dyst_coords.shape)

            dyst_coords_samples = np.array(dyst_coords_samples)
            print(dyst_coords_samples.shape)

            # Plot the univariate timeseries after augmentation
            plot_trajs_univariate(
                dyst_coords_samples, 
                selected_dim = 1,
                save_dir = "tests/figs", 
                plot_name = f"{dyst_name}_univariate_{augmentation_cls_name}"
            )

            plot_trajs_multivariate(
                dyst_coords_samples, 
                save_dir = "tests/figs", 
                plot_name = f"{dyst_name}_{augmentation_cls_name}"
            )
    return gts_datasets_dict



if __name__ == "__main__":
    # NOTE: augmentations so far are only on univariate trajectories
    parser = argparse.ArgumentParser()
    parser.add_argument("dysts_names", help="Name of the dynamical system", nargs="+", type=str)
    args = parser.parse_args()

    dysts_names = args.dysts_names

    print("Applying system-scale transformations")
    gts_datasets_dict = apply_augmentations_system(dysts_names)
    assert list(gts_datasets_dict.keys()) == dysts_names, "Mismatch in dyst names"

    print("Applying ensemble-scale transformations")
    print("Avaialble data files for ensemble-scale transform: ", gts_datasets_dict)
    # for every ensemble-scale augmentation
    for augmentation_cls_name in augmentation_cls_dict["ensemble_scale"]:
        print(augmentation_cls_name)
        augmentation_cls = getattr(augmentations_module, augmentation_cls_name)
        print("Applying ensemble-scale augmentation: ", augmentation_cls.__name__)
        kwargs = augmentation_cls_kwargs[augmentation_cls_name]
        print("kwargs: ", kwargs)

        # for every pair of dysts
        for i, j in sample_index_pairs(len(dysts_names), num_pairs=1):
            dyst1, dyst2 = dysts_names[i], dysts_names[j]
            print(f"Applying ensemble-scale augmentation to {dyst1} and {dyst2}")
            num_samples = min(len(gts_datasets_dict[dyst1]), len(gts_datasets_dict[dyst2]))
            
            dyst_pair_coords_samples = []
            for sample_idx in range(num_samples):
                print(f"Augmenting sample index {sample_idx}")
                gts_dataset = partial(
                    augmentation_cls,
                    **kwargs,
                )(
                    gts_datasets_dict[dyst1][sample_idx], 
                    gts_datasets_dict[dyst2][sample_idx]
                )

                dyst_pair_coords, _ = stack_and_extract_metadata(gts_dataset)
                dyst_pair_coords_samples.append(dyst_pair_coords)
                print("augmented data shape: ", dyst_pair_coords.shape)

            dyst_pair_coords_samples = np.array(dyst_pair_coords_samples)
            print(dyst_pair_coords_samples.shape)

            # Plot the univariate timeseries after ensemble-scale augmentation
            plot_trajs_univariate(
                dyst_pair_coords_samples, 
                selected_dim = 1,
                plot_name=f"{'_'.join([dyst1, dyst2])}_univariate_{augmentation_cls_name}"
            )
            plot_trajs_multivariate(
                dyst_pair_coords_samples, 
                save_dir = "tests/figs", 
                plot_name = f"{'_'.join([dyst1, dyst2])}_{augmentation_cls_name}"
            )