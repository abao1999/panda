import os
import numpy as np

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.dataset as ds

import matplotlib.pyplot as plt
import argparse


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

    # read arrow file into pandas dataframe
    dyst_df = read_arrow_ds(filepath)
    print(dyst_df.columns)

    # print(dyst_df['target'][0].shape) # this confirms that gluonts in writing the arrow file concatenates dimensions i.e. (1024, 3) -> (3073,)

    # NOTE: we are assuming no jagged arrays i.e. every row of target._np_shape is the same
    target_shape = dyst_df['target._np_shape'][0]
    print(target_shape)
    assert len(target_shape) <= 2, "too many dimensions!"

    # stack and reshape
    dyst_data = np.stack(dyst_df['target'], axis=0).reshape(-1, target_shape[0], target_shape[1])
    assert dyst_data.shape[0] == dyst_df.shape[0], "shapes are messed up"
    
    print(type(dyst_data))
    print(dyst_data.shape)

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



# # ================== Could be useful, but will prob eventually move into utils or tests =====================
# # TODO: once we add initial condition option for dysts.make_trajectory_ensemble, we need to modify this
# def make_single_dyst(
#         dyst_name: str = "Lorenz", 
#         split: str = "train",
#         num_points: int = 1024,
#         num_periods: int = 5,
# ) -> None:
#     """
#     A test function to make a single [dyst_name].arrow file in data/train split
#     Directly calls dysts.flows.[dyst_name].make_trajectory where dyst_name is the name of the dyst class
#     Samples initial conditions by integrating forward an initial trajectory and sampling points from it uniformly
#     Thus, initial conditions are "on attractor" (see shadowing lemma)

#     NOTE: this should perform similar functionality to make_single_dyst_from_ensemble but could be useful for debugging
#     """

#     # set up save directory
#     data_dir = os.path.join(WORK_DIR, 'data', split)
#     os.makedirs(data_dir, exist_ok=True)

#     # get dysts class associated with dyst_name
#     dyst_module = importlib.import_module("dysts.flows")
#     dyst_class_ = getattr(dyst_module, dyst_name)
#     print(dyst_class_)
    
#     # make trajectory
#     traj = dyst_class_().make_trajectory(num_points, standardize=True, pts_per_period=num_points//num_periods)

#     # TODO: sample initial conditions

#     # save trajectories to arrow file
#     convert_to_arrow(os.path.join(data_dir, f"{dyst_name}.arrow"), traj)