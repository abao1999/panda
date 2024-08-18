import os
import numpy as np

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.dataset as ds

import matplotlib.pyplot as plt
import argparse

from gluonts.dataset.common import FileDataset, ListDataset
from pathlib import Path

from chronos_dysts.augmentations import stack_and_extract_metadata
from typing import List

WORK_DIR = os.getenv('WORK')

def get_dyst_filepaths(dyst_name: str) -> List[str]:
    """ 
    [dyst_name].arrow could either be in data/train or data/test
    Check if [dyst_name].arrow is in either data/train or data/test
    """
    possible_train_dir = os.path.join(WORK_DIR, 'data/train', dyst_name)
    possible_test_dir = os.path.join(WORK_DIR, 'data/test', dyst_name)

    if os.path.exists(possible_train_dir):
        dyst_dir = possible_train_dir
    elif os.path.exists(possible_test_dir):
        dyst_dir = possible_test_dir
    else:
        raise Exception(f"Directory {dyst_name} does not exist in data/train or data/test.")

    print(f"Found dyst directory: {dyst_dir}")
    filepaths = list(Path(dyst_dir).glob("*.arrow"))
    print(f"Found {len(filepaths)} files in {dyst_dir}")
    return filepaths

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

def read_arrow_direct(filepath):
    """
    Read data directly from ArrowFile, default reader is ipc.RecordBatchFileReader
    """
    # Open the Arrow file
    with pa.memory_map(filepath, 'r') as source:
        # Initialize the ArrowFile reader
        reader = ipc.RecordBatchFileReader(source)
        # Iterate through the RecordBatches in the file
        for i in range(reader.num_record_batches):
            # Read each RecordBatch
            batch = reader.get_record_batch(i)
            # Convert RecordBatch to a PyArrow Table
            table = pa.Table.from_batches([batch])
            # Convert Table to a Pandas DataFrame if needed
            df = table.to_pandas()
            # Display the DataFrame
            print(df.head())

    return df


def plot_trajs_multivariate(dyst_data, plot_name=None):
    num_ics = dyst_data.shape[0]
    num_ics_to_plot = 5 if num_ics > 5 else num_ics
    if plot_name is None:
        plot_name = "dyst"
    # Plot the trajectories
    plt.figure(figsize=(6,6))
    for ic_idx in range(num_ics_to_plot):
        # plot x and y
        plt.plot(dyst_data[ic_idx, 0, :], dyst_data[ic_idx, 1, :], alpha=0.5, linewidth=1)
        plt.scatter(*dyst_data[ic_idx, :2, 0], marker="*", s=100, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(plot_name)
    plt.savefig(f"tests/{plot_name}.png", dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for ic_idx in range(num_ics_to_plot):
        # plot x and y and z
        ax.plot(dyst_data[ic_idx, 0, :], dyst_data[ic_idx, 1, :], dyst_data[ic_idx, 2, :], alpha=0.5, linewidth=1)  # X,Y,Z
        ax.scatter(*dyst_data[ic_idx, :, 0], marker="*", s=100, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    plt.title(plot_name)
    plt.savefig(f"tests/{plot_name}_3D.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dyst_name", help="Name of the dynamical system", type=str)
    args = parser.parse_args()

    filepaths = get_dyst_filepaths(args.dyst_name)

    dyst_coords_samples = []
    for filepath in filepaths:
        # # read arrow file into pandas dataframe
        # dyst_df = read_arrow_ds(filepath)
        # print(dyst_df.columns)
        # # stack
        # dyst_coords = np.stack(dyst_df['target'], axis=0)
        # print(dyst_coords.shape)

        # create dataset by reading directly from filepath into FileDataset
        gts_dataset = FileDataset(path=Path(filepath), freq="h", one_dim_target=True) # TODO: consider other frequencies?

        # extract the coordinates
        dyst_coords, metadata = stack_and_extract_metadata(gts_dataset)
        dyst_coords_samples.append(dyst_coords)

        print("original data shape: ", dyst_coords.shape)

    dyst_coords_samples = np.array(dyst_coords_samples)
    print(dyst_coords_samples.shape)

    # # plot the trajectories
    plot_trajs_multivariate(dyst_coords_samples, plot_name=args.dyst_name)