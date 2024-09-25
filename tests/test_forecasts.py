import numpy as np
import os
import argparse
from pathlib import Path

from dystformer.utils import (
    get_dyst_filepaths,
    stack_and_extract_metadata,
    plot_trajs_multivariate,
    plot_forecast_trajs_multivariate,
    plot_forecast_gt_trajs_multivariate,
)
from gluonts.dataset.common import FileDataset


WORK_DIR = os.getenv("WORK", '')
FORECAST_DATA_DIR = os.path.join(WORK_DIR, "data/forecasts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dyst_name", help="Name of the dynamical system", type=str)
    args = parser.parse_args()

    dyst_name = args.dyst_name
    gt_dyst_filepaths = get_dyst_filepaths(dyst_name)
    num_samples_gt = len(gt_dyst_filepaths)
    print(f"Found {num_samples_gt} files for {dyst_name}")

    forecast_dyst_dir = os.path.join(FORECAST_DATA_DIR, dyst_name)
    if os.path.exists(forecast_dyst_dir):
        print(f"Found dyst forecast directory: {forecast_dyst_dir}")
        fc_dyst_filepaths = sorted(list(Path(forecast_dyst_dir).glob("*.npy")), key=lambda x: int(x.stem.split("_")[0]))
        num_samples_forecast = len(fc_dyst_filepaths)
        print(f"Found {num_samples_forecast} files in {forecast_dyst_dir}")
    else:
        raise Exception(f"Directory {forecast_dyst_dir} does not exist in {FORECAST_DATA_DIR}")

    num_samples = min(num_samples_gt, num_samples_forecast)
    gt_dyst_filepaths = gt_dyst_filepaths[:num_samples]
    fc_dyst_filepaths = fc_dyst_filepaths[:num_samples]
    
    gt_stems = [Path(filepath).stem for filepath in gt_dyst_filepaths]
    fc_stems = [Path(filepath).stem for filepath in fc_dyst_filepaths]

    if gt_stems == fc_stems:
        print("All filepath stems match element-wise")
    else:
        print("Filepath stems do not match element-wise")

    # accumulate the forecasted trajectories coordinates across samples
    print("Accumulating forecasted trajectories")
    fc_dyst_coords_samples = []
    for filepath in fc_dyst_filepaths:
        # load the forecasted trajectories
        print(f"Loading {filepath}")
        # shape of data is (dim, num_samples=20, prediction_length)
        dyst_coords = np.load(filepath)
        print("data shape: ", dyst_coords.shape)
        # dyst_coords = dyst_coords[:, 0, :].squeeze() # just take the first samples
        dyst_coords = np.mean(dyst_coords, axis=1) # average along axis=1 to get (dim, prediction_length)
        # dyst_coords should now be shape (dim, prediction_length)
        fc_dyst_coords_samples.append(dyst_coords)

    # should be shape (num_samples, dim, prediction_length)
    fc_dyst_coords_samples = np.array(fc_dyst_coords_samples)
    print(fc_dyst_coords_samples.shape)

    # accumulate the ground truth trajectories coordinates across samples
    print("Accumulating ground truth trajectories")
    gt_dyst_coords_samples = []
    for filepath in gt_dyst_filepaths:
        print("Loading ", filepath)
        # create dataset by reading directly from filepath into FileDataset
        gts_dataset = FileDataset(path=Path(filepath), freq="h", one_dim_target=True) # TODO: consider other frequencies?
        # extract the coordinates
        dyst_coords, metadata = stack_and_extract_metadata(gts_dataset)
        print("data shape: ", dyst_coords.shape)
        gt_dyst_coords_samples.append(dyst_coords)

    # should be shape (num_samples, dim, prediction_length)
    gt_dyst_coords_samples = np.array(gt_dyst_coords_samples)
    print(gt_dyst_coords_samples.shape)

    dim = gt_dyst_coords_samples.shape[1]
    npts = gt_dyst_coords_samples.shape[2]
    assert dim == fc_dyst_coords_samples.shape[1], "Mismatch in dimensions of ground truth and forecasted trajectories"

    context_length = npts - fc_dyst_coords_samples.shape[2]
    assert context_length > 0, "Context length must be greater than 0"
    print("Assuming context length: ", context_length)

    # modify fc_dyst_coords_samples by extending to include first context_length points from gt_dyst_coords_samples
    print(f"Modifying forecasted trajectories to prepend {context_length} context points from ground truth")
    full_fc_dyst_coords_samples = np.zeros((num_samples, dim, npts))
    for i in range(num_samples):
        # for this specific sample
        gt_coords = gt_dyst_coords_samples[i]
        fc_coords = fc_dyst_coords_samples[i]
        print(gt_coords.shape, fc_coords.shape)
        # horizontally stack the first context_length points from gt_coords with fc_coords
        full_fc_dyst_coords_samples[i] = np.hstack((gt_coords[:, :context_length], fc_coords))

    print(full_fc_dyst_coords_samples.shape)

    # # plot the forecasted trajectories
    # plot_forecast_trajs_multivariate(
    #     full_fc_dyst_coords_samples, 
    #     context_length=context_length,
    #     save_dir="eval_results/figs", 
    #     plot_name=f"{dyst_name}_forecast",
    #     num_samples_to_plot=4,
    # )

    # plot the forecasted and ground truth trajectories
    plot_forecast_gt_trajs_multivariate(
        full_fc_dyst_coords_samples,
        gt_dyst_coords_samples,
        context_length=context_length,
        save_dir="eval_results/figs",
        plot_name=f"{dyst_name}_forecast_gt",
        num_samples_to_plot=4,
    )