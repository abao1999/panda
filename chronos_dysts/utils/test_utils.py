import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt


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


# Plotting utils
def plot_trajs_univariate(dyst_data, selected_dim=0, save_dir="tests/figs", plot_name=None):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = dyst_data.shape[0]
    # limit plotting to at most 5 samples
    num_samples_to_plot = 5 if num_samples > 5 else num_samples
    if plot_name is None:
        plot_name = "dyst"

    # Plot the first coordinate
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    print("Plotting 2D trajectories and saving to ", save_path)

    plt.figure(figsize=(6,6))
    for sample_idx in range(num_samples_to_plot):
        plt.plot(dyst_data[sample_idx, selected_dim, :], alpha=0.5, linewidth=1)
    plt.xlabel('timesteps')
    plt.title(plot_name)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_trajs_multivariate(dyst_data, save_dir="tests/figs", plot_name=None):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = dyst_data.shape[0]
    # limit plotting to at most 5 samples
    num_samples_to_plot = 5 if num_samples > 5 else num_samples
    if plot_name is None:
        plot_name = "dyst"

    # Plot the first two coordinates
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    print("Plotting 2D trajectories and saving to ", save_path)
    plt.figure(figsize=(6,6))
    for sample_idx in range(num_samples_to_plot):
        # plot x and y
        plt.plot(dyst_data[sample_idx, 0, :], dyst_data[sample_idx, 1, :], alpha=0.5, linewidth=1)
        plt.scatter(*dyst_data[sample_idx, :2, 0], marker="*", s=100, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(plot_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

    # 3D plot (first three coordinates)
    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    print("Plotting 3D trajectories and saving to ", save_path)
    if dyst_data.shape[1] >= 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        for sample_idx in range(num_samples_to_plot):
            # plot x and y and z
            ax.plot(dyst_data[sample_idx, 0, :], dyst_data[sample_idx, 1, :], dyst_data[sample_idx, 2, :], alpha=0.5, linewidth=1)  # X,Y,Z
            ax.scatter(*dyst_data[sample_idx, :3, 0], marker="*", s=100, alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        plt.title(plot_name)
        plt.savefig(save_path, dpi=300)
        plt.close()




def read_arrow_ds(filepath):
    """
    Read data using pyarrow dataset
    Note: not used currently but could be useful util in the future
    """
    # imports only needed for this function
    import pyarrow.dataset as ds

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
    Note: not used currently but could be useful util in the future
    """
    # imports only needed for this function
    import pyarrow as pa
    import pyarrow.ipc as ipc

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