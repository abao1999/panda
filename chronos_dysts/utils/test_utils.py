import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# for type hints
from typing import Optional, List, Dict, Callable
from gluonts.dataset import Dataset

from chronos_dysts.utils import stack_and_extract_metadata

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
    filepaths = sorted(list(Path(dyst_dir).glob("*.arrow")), key=lambda x: int(x.stem.split("_")[0]))
    print(f"Found {len(filepaths)} files in {dyst_dir}")
    return filepaths


from collections import defaultdict
from gluonts.dataset.common import FileDataset
def get_dyst_datasets(dyst_name: str) -> List[Dataset]:
    """
    Returns list of datasets associated with dyst_name, converted to FileDataset
    """
    filepaths = get_dyst_filepaths(dyst_name)
    
    gts_datasets_list = []
    # for every file in the directory
    for filepath in filepaths:
        # create dataset by reading directly from filepath into FileDataset
        gts_dataset = FileDataset(path=Path(filepath), freq="h", one_dim_target=True) # TODO: consider other frequencies?
        gts_datasets_list.append(gts_dataset)
    return gts_datasets_list

def get_dysts_datasets_dict(dysts_names: List[str]) -> Dict[str, List[Dataset]]:
    """
    Returns a dictionary with key as dyst_name and value as list of FileDatasets loaded from that dyst_name folder
    """
    gts_datasets_dict = defaultdict(list)
    for dyst_name in dysts_names:
        gts_datasets_dict[dyst_name] = get_dyst_datasets(dyst_name)
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
        dyst_coords, _ = stack_and_extract_metadata(gts_dataset)
        dyst_coords_samples.append(dyst_coords)
        print("data shape: ", dyst_coords.shape)

    dyst_coords_samples = np.array(dyst_coords_samples)
    print(dyst_coords_samples.shape)
    return dyst_coords_samples


# Plotting utils
def plot_trajs_univariate(
        dyst_data, 
        selected_dim=0, 
        save_dir="tests/figs", 
        plot_name=None,
        num_samples_to_plot=None,
) -> None:
    """
    Plot univariate timeseries from dyst_data
    """
    os.makedirs(save_dir, exist_ok=True)
    num_samples = dyst_data.shape[0]
    if num_samples_to_plot is None:
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
    plt.title(plot_name.replace('_', ' '))
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_trajs_multivariate(
        dyst_data, 
        save_dir="tests/figs", 
        plot_name=None, 
        num_samples_to_plot=None,
) -> None:
    """
    Plot multivariate timeseries from dyst_data
    """
    os.makedirs(save_dir, exist_ok=True)
    num_samples = dyst_data.shape[0]
    if num_samples_to_plot is None:
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
    plt.title(plot_name.replace('_', ' '))
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
        plt.title(plot_name.replace('_', ' '))
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_forecast_trajs_multivariate(
        dyst_data, 
        context_length,
        save_dir="tests/figs", 
        plot_name=None, 
        num_samples_to_plot=None,
) -> None:
    """
    Plot multivariate timeseries from dyst_data
    """
    dyst_name = plot_name.split('_')[0]
    print("Plotting forecast vs ground truth for ", dyst_name)
    os.makedirs(save_dir, exist_ok=True)
    num_samples = dyst_data.shape[0]
    if num_samples_to_plot is None:
        # limit plotting to at most 5 samples
        num_samples_to_plot = 5 if num_samples > 5 else num_samples
    if plot_name is None:
        plot_name = "dyst"

    colors = plt.rcParams['axes.prop_cycle'] 
    colors = colors.by_key()['color']

    # Plot the first two coordinates
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    print("Plotting 2D trajectories and saving to ", save_path)
    plt.figure(figsize=(6,6))
    for sample_idx in range(num_samples_to_plot):
        plt.scatter(*dyst_data[sample_idx, :2, 0], marker="*", s=25, alpha=1, color=colors[sample_idx])
        # plot x and y
        plt.plot(
            dyst_data[sample_idx, 0, :context_length], 
            dyst_data[sample_idx, 1, :context_length], 
            alpha=0.25, 
            linewidth=1,
            color=colors[sample_idx],
        )
        plt.scatter(*dyst_data[sample_idx, :2, context_length], marker="*", s=100, alpha=1, color=colors[sample_idx])
        # plot x and y
        plt.plot(
            dyst_data[sample_idx, 0, context_length:], 
            dyst_data[sample_idx, 1, context_length:], 
            alpha=0.5, 
            linewidth=2,
            color=colors[sample_idx],
        )
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{dyst_name} Forecast")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # 3D plot (first three coordinates)
    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    print("Plotting 3D trajectories and saving to ", save_path)
    if dyst_data.shape[1] >= 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        for sample_idx in range(num_samples_to_plot):
            ax.scatter(*dyst_data[sample_idx, :3, 0], marker="*", s=25, alpha=1, color=colors[sample_idx])
            # plot x and y and z
            ax.plot(
                dyst_data[sample_idx, 0, :context_length], 
                dyst_data[sample_idx, 1, :context_length], 
                dyst_data[sample_idx, 2, :context_length], 
                alpha=0.5, 
                linewidth=1,
                color=colors[sample_idx],
            )
            ax.scatter(*dyst_data[sample_idx, :3, context_length], marker="*", s=100, alpha=1, color=colors[sample_idx])
            ax.plot(
                dyst_data[sample_idx, 0, context_length:], 
                dyst_data[sample_idx, 1, context_length:], 
                dyst_data[sample_idx, 2, context_length:], 
                alpha=0.5, 
                linewidth=2,
                color=colors[sample_idx],
            )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        plt.title(f"{dyst_name} Forecast")
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_forecast_gt_trajs_multivariate(
        fc_data,
        gt_data, 
        context_length,
        save_dir="tests/figs", 
        plot_name=None, 
        num_samples_to_plot=None,
) -> None:
    """
    Plot multivariate timeseries from ground trugh and forecasted data
    """
    dyst_name = plot_name.split('_')[0]
    print("Plotting forecast vs ground truth for ", dyst_name)
    os.makedirs(save_dir, exist_ok=True)
    num_samples = gt_data.shape[0]
    assert num_samples == fc_data.shape[0], "Mismatch in number of samples"
    if num_samples_to_plot is None:
        # limit plotting to at most 5 samples
        num_samples_to_plot = 5 if num_samples > 5 else num_samples
    if plot_name is None:
        plot_name = "dyst"

    colors = plt.rcParams['axes.prop_cycle'] 
    colors = colors.by_key()['color']

    # Plot the first two coordinates
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    print("Plotting 2D trajectories and saving to ", save_path)
    plt.figure(figsize=(6,6))
    for sample_idx in range(num_samples_to_plot):
        plt.scatter(*gt_data[sample_idx, :2, 0], marker="*", s=25, alpha=1, color=colors[sample_idx])
        # plot x and y
        plt.plot(
            gt_data[sample_idx, 0, :], 
            gt_data[sample_idx, 1, :], 
            alpha=0.25, 
            linewidth=1,
            color=colors[sample_idx],
            label="Ground Truth",
        )
        plt.scatter(*gt_data[sample_idx, :2, context_length], marker="*", s=100, alpha=1, color=colors[sample_idx])
        # plot x and y
        plt.plot(
            fc_data[sample_idx, 0, context_length:], 
            fc_data[sample_idx, 1, context_length:], 
            alpha=0.5, 
            linewidth=1,
            linestyle='dashed',  # Set the linestyle to dashed
            color=colors[sample_idx],
            label="Forecast",
        )
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{dyst_name} Forecast vs Ground Truth")
    # plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # 3D plot (first three coordinates)
    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    print("Plotting 3D trajectories and saving to ", save_path)
    if gt_data.shape[1] >= 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        for sample_idx in range(num_samples_to_plot):
            ax.scatter(*gt_data[sample_idx, :3, 0], marker="*", s=25, alpha=1, color=colors[sample_idx])
            # plot x and y and z
            ax.plot(
                gt_data[sample_idx, 0, :], 
                gt_data[sample_idx, 1, :], 
                gt_data[sample_idx, 2, :], 
                alpha=0.5, 
                linewidth=1,
                color=colors[sample_idx],
                label="Ground Truth",
            )
            ax.scatter(*gt_data[sample_idx, :3, context_length], marker="*", s=100, alpha=1, color=colors[sample_idx])
            ax.plot(
                fc_data[sample_idx, 0, context_length:], 
                fc_data[sample_idx, 1, context_length:], 
                fc_data[sample_idx, 2, context_length:], 
                alpha=0.5, 
                linewidth=1,
                linestyle='dashed',  # Set the linestyle to dashed
                color=colors[sample_idx],
                label="Forecast",
            )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        plt.title(f"{dyst_name} Forecast vs Ground Truth")
        # plt.legend()
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