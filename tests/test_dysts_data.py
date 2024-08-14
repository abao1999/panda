import os
import numpy as np

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.dataset as ds

import matplotlib.pyplot as plt
import argparse


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


def plot_trajs(dyst_data):
    # Plot the trajectories
    plt.figure(figsize=(6,6))
    for ic_idx in range(5):
        plt.plot(dyst_data[ic_idx, :, 0], dyst_data[ic_idx, :, 1], alpha=0.5, linewidth=1)
        plt.scatter(*dyst_data[ic_idx, 0, :2], marker="*", s=100, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{args.dyst_name} Y and X")
    plt.savefig(f"tests/{args.dyst_name}.png", dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for ic_idx in range(5):
        ax.plot(dyst_data[ic_idx, : , 0], dyst_data[ic_idx, :, 1], dyst_data[ic_idx, :, 2], alpha=0.5, linewidth=1)  # X,Y,Z
        ax.scatter(*dyst_data[ic_idx, 0, :], marker="*", s=100, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    plt.title(args.dyst_name)
    plt.savefig(f"tests/{args.dyst_name}_3D.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dyst_name", help="Name of the dynamical system", type=str)
    args = parser.parse_args()
    filepath = get_dyst_filepath(args.dyst_name)
    # with ipc.open_file(filepath) as reader:
    #     df = reader.read_pandas()

    # read arrow file into pandas dataframe
    dyst_df = read_arrow_ds(filepath)
    print(dyst_df.columns)

    # print(dyst_df['target'][0].shape) # this confirms that gluonts in writing the arrow file concatenates dimensions i.e. (1024, 3) -> (3073,)

    # stack
    dyst_data = np.stack(dyst_df['target'], axis=0)

    # if we have 3 dimensions
    if 'target._np_shape' in dyst_df.columns:
        # NOTE: we are assuming no jagged arrays i.e. every row of target._np_shape is the same
        target_shape = dyst_df['target._np_shape'][0]
        print(target_shape)
        assert len(target_shape) == 2, "incorrect number of dimensions!"
        # reshape
        dyst_data = dyst_data.reshape(-1, target_shape[0], target_shape[1])

    assert dyst_data.shape[0] == dyst_df.shape[0], "shapes are messed up"
    
    print(type(dyst_data))
    print(dyst_data.shape)

    # # plot the trajectories
    # plot_trajs(dyst_data)