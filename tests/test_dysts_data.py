import os
import numpy as np

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.dataset as ds

import matplotlib.pyplot as plt
import argparse

# TODO: once we add initial condition option for dysts.make_trajectory_ensemble, we need to modify this

def get_dyst_filepath(dyst_name):
    """ 
    [dyst_name].arrow could either be in data/train or data/test
    Check if [dyst_name].arrow is in either data/train or data/test
    """
    work_dir = os.getenv('WORK')
    filepath_train = os.path.join(work_dir, 'data/train', f"{dyst_name}.arrow")
    filepath_test = os.path.join(work_dir, 'data/test', f"{dyst_name}.arrow")

    if os.path.exists(filepath_train):
        filepath = filepath_train
    elif os.path.exists(filepath_test):
        filepath = filepath_test
    else:
        train_dir = os.path.join(work_dir, 'data/train')
        test_dir = os.path.join(work_dir, 'data/test')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dyst_name", help="Name of the dynamical system", type=str)
    args = parser.parse_args()
    filepath = get_dyst_filepath(args.dyst_name)
    # with ipc.open_file(filepath) as reader:
    #     df = reader.read_pandas()
    dyst_df = read_arrow_ds(filepath)
    print(dyst_df.columns)
    dyst_data = np.stack(dyst_df['target'].to_numpy())
    print(type(dyst_data))
    print(dyst_data.shape)

    # Plot the data
    plt.figure(figsize=(6,6))
    plt.plot(dyst_data[:, 0], dyst_data[:, 1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{args.dyst_name} Y and X")
    plt.savefig(f"tests/{args.dyst_name}.png", dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(dyst_data[:, 0], dyst_data[:, 1], dyst_data[:, 2], linewidth=2)  # X,Y,Z
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    plt.title(args.dyst_name)
    plt.savefig(f"tests/{args.dyst_name}_3D.png", dpi=300)
    plt.close()