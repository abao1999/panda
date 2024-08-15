import os
import numpy as np

import pyarrow.dataset as ds

import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from chronos_dysts.augmentations import (
    stack_and_extract_metadata,
    SystemTransform,
    IdentityTransform,
    RandomConvexCombinationTransform,
    RandomAffineTransform,
    RandomProjectedSkewTransform,
)

from gluonts.dataset.common import FileDataset, ListDataset

WORK_DIR = os.getenv('WORK')

def get_dyst_filepath(dyst_name):
    """ 
    [dyst_name].arrow could either be in data/train or data/test
    Check if [dyst_name].arrow is in either data/train or data/test
    """
    filepath_train = os.path.join(WORK_DIR, 'data/train', f"{dyst_name}.arrow")
    filepath_test = os.path.join(WORK_DIR, 'data/test', f"{dyst_name}.arrow")

    if os.path.exists(filepath_train):
        filepath = filepath_train
    elif os.path.exists(filepath_test):
        filepath = filepath_test
    else:
        train_dir = os.path.join(WORK_DIR, 'data/train')
        test_dir = os.path.join(WORK_DIR, 'data/test')
        available_files = []
        
        if os.path.exists(train_dir):
            available_files += os.listdir(train_dir)
        
        if os.path.exists(test_dir):
            available_files += os.listdir(test_dir)
        
        raise FileNotFoundError(f"File {dyst_name}.arrow does not exist. Available files: {available_files}")

    print(f"Found Arrow file: {filepath}")
    return filepath

def read_arrow_ds(filepath):
    """
    Read data using pyarrow dataset
    """
    # Load the dataset from the Arrow file
    dataset = ds.dataset(filepath, format="arrow")
    # Convert the dataset to a Table
    table = dataset.to_table()
    # Convert the Table to a Pandas DataFrame if needed
    df = table.to_pandas()
    # Display the DataFrame
    print(df.head())
    return df


def plot_trajs_univariate(dyst_data, plot_name=None):
    num_ics = dyst_data.shape[0]
    num_ics_to_plot = 5 if num_ics > 5 else num_ics
    if plot_name is None:
        plot_name = "dyst"
    # Plot the trajectories
    plt.figure(figsize=(6,6))
    for ic_idx in range(num_ics_to_plot):
        # plot x and y
        plt.plot(dyst_data[ic_idx, :], alpha=0.5, linewidth=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(plot_name)
    plt.savefig(f"tests/{plot_name}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # NOTE: augmentations so far are only on univariate trajectories
    parser = argparse.ArgumentParser()
    parser.add_argument("dysts_names", help="Name of the dynamical system", nargs="+", type=str)
    args = parser.parse_args()

    dysts_names = args.dysts_names
    print(dysts_names)

    num_systems = len(dysts_names)
    gts_datasets_list = []
    for i in range(num_systems):

        # load dyst data from corresponding filepath
        dyst_name = dysts_names[i]
        filepath = get_dyst_filepath(dyst_name)

        # create dataset by reading directly from filepath into FileDataset
        gts_dataset = FileDataset(path=Path(filepath), freq="h", one_dim_target=True) # TODO: consider other frequencies?
        # print(vars(gts_dataset))
        gts_datasets_list.append(gts_dataset) # save for ensemble-scale transformations

        # extract the coordinates
        coords, metadata = stack_and_extract_metadata(gts_dataset)
        print("original data shape: ", coords.shape)

        # Apply augmentation, which takes in GluonTS Dataset and returns ListDataset
        # augmentation = RandomAffineTransform(out_dim=10, scale=1e-2, random_seed=0)
        augmentation = RandomConvexCombinationTransform(num_combinations=100, alpha=0.6, random_seed=0)
        augmentation_class_name = augmentation.__class__.__name__
        print(augmentation_class_name)
        gts_dataset = augmentation(gts_dataset)

        # extract the coordinates
        coords, _ = stack_and_extract_metadata(gts_dataset)
        print("augmented data shape: ", coords.shape)

        # Plot the univariate timeseries after augmentation
        plot_trajs_univariate(coords, plot_name=f"{dyst_name}_{augmentation_class_name}")

    # if doing ensemble scale transformation, such as RandomProjectedSkewTransform
    ensemble_augmentation = RandomProjectedSkewTransform(num_skew_pairs=5, embedding_dim=10, scale=1e-2, random_seed=0)
    augmentation_class_name = ensemble_augmentation.__class__.__name__
    print(augmentation_class_name)
    # RandomProjectedSkewTransform returns a dict of ListDatasets
    gts_datasets_dict = ensemble_augmentation(gts_datasets_list)
    print(gts_datasets_dict.keys())
    for dyst_pair_ids, gts_dataset in gts_datasets_dict.items():
        dyst_pair = [dysts_names[i] for i in dyst_pair_ids]
        coords, _ = stack_and_extract_metadata(gts_dataset)
        print("augmented data shape: ", coords.shape)
        # Plot the univariate timeseries after ensemble-scale augmentation
        plot_trajs_univariate(coords, plot_name=f"{'_'.join(dyst_pair)}_{augmentation_class_name}")
