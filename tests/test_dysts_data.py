import numpy as np

import argparse

from gluonts.dataset.common import FileDataset, ListDataset
from pathlib import Path

from chronos_dysts.augmentations import stack_and_extract_metadata
from chronos_dysts.utils import get_dyst_filepaths, plot_trajs_multivariate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dyst_name", help="Name of the dynamical system", type=str)
    args = parser.parse_args()

    filepaths = get_dyst_filepaths(args.dyst_name)

    dyst_coords_samples = []
    for filepath in filepaths:
        # create dataset by reading directly from filepath into FileDataset
        gts_dataset = FileDataset(path=Path(filepath), freq="h", one_dim_target=True) # TODO: consider other frequencies?

        # extract the coordinates
        dyst_coords, metadata = stack_and_extract_metadata(gts_dataset)
        dyst_coords_samples.append(dyst_coords)

        print("data shape: ", dyst_coords.shape)

    dyst_coords_samples = np.array(dyst_coords_samples)
    print(dyst_coords_samples.shape)

    # plot the trajectories
    plot_trajs_multivariate(dyst_coords_samples, save_dir="tests/figs", plot_name=args.dyst_name)