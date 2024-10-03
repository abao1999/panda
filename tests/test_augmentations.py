import argparse
from functools import partial
from typing import List

import numpy as np

from dystformer import augmentations
from dystformer.utils import (
    accumulate_dyst_samples,
    get_dysts_datasets_dict,
    plot_trajs_multivariate,
    sample_index_pairs,
    stack_and_extract_metadata,
)

# list of augmentation classes to apply
AUG_CLS_DICT = {
    "system_scale": ["RandomConvexCombinationTransform", "RandomAffineTransform"],
    "ensemble_scale": ["RandomProjectedSkewTransform"],
}
# kwargs associated with each augmentation class
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


def apply_augmentations_system(
    dysts_names: List[str], split: str = "train", one_dim_target: bool = True
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
            augmentation_fn = partial(augmentation_cls, **kwargs)
            # accumulate coords in sample dimension, while applying augmentation to each coords
            dyst_coords_samples = accumulate_dyst_samples(
                dyst_name,
                gts_datasets_dict,
                augmentation_fn,
            )

            # # Plot the univariate timeseries after augmentation
            # plot_trajs_univariate(
            #     dyst_coords_samples,
            #     selected_dim=1,
            #     save_dir="tests/figs",
            #     plot_name=f"{dyst_name}_univariate_{augmentation_cls_name}",
            # )

            plot_trajs_multivariate(
                dyst_coords_samples,
                save_dir="tests/figs",
                plot_name=f"{dyst_name}_{augmentation_cls_name}",
            )


def apply_augmentations_ensemble(
    dysts_names: List[str],
    split: str = "train",
    num_pairs_dysts: int = 1,
    one_dim_target: bool = True,
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
        augmentation_fn = partial(augmentation_cls, **kwargs)

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

            # Plot the univariate timeseries after ensemble-scale augmentation
            # plot_trajs_univariate(
            #     dyst_pair_coords_samples,
            #     selected_dim=1,
            #     plot_name=f"{'_'.join([dyst1, dyst2])}_univariate_{augmentation_cls_name}",
            # )
            plot_trajs_multivariate(
                dyst_pair_coords_samples,
                save_dir="tests/figs",
                plot_name=f"{'_'.join([dyst1, dyst2])}_{augmentation_cls_name}",
            )


if __name__ == "__main__":
    # NOTE: augmentations so far are only on univariate trajectories
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
