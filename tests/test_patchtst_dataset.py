import os
from functools import partial
from pathlib import Path

import hydra
import numpy as np
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.utils import (
    has_enough_observations,
)


def test_patchtst_dataset(cfg):
    # Get list of files to use for training
    # add all files in train_data_dir to train_data_paths, a list of arrow data filepaths

    # Get the path for "$WORK/data/train/Lorenz"
    train_data_dir = os.path.expandvars("$WORK/data/train/")
    train_data_paths = [
        os.path.join(train_data_dir, "Lorenz/0_T-1024.arrow"),
        os.path.join(train_data_dir, "Lorenz96/0_T-1024.arrow"),
    ]
    # if cfg.train_data_dir is not None:
    #     train_data_paths = list(
    #         filter(lambda file: file.is_file(), Path(cfg.train_data_dir).rglob("*"))
    #     )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=cfg.min_past + cfg.patchtst.prediction_length,
                max_missing_prop=cfg.max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h", one_dim_target=False),  # type: ignore
        )
        for data_path in train_data_paths
    ]

    # set probabilities (how we weight draws from each data file)
    if isinstance(cfg.probability, float):
        probability = cfg.probability
    elif cfg.probability is None:
        probability = [1.0 / len(train_datasets)] * len(train_datasets)
    assert isinstance(probability, list)

    assert len(train_datasets) == len(probability)

    dataset = PatchTSTDataset(
        datasets=train_datasets,
        probabilities=probability,
        context_length=cfg.patchtst.context_length,
        prediction_length=cfg.patchtst.prediction_length,
        mode="train",
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    for data in dataset:
        print(data)
        print(data["future_values"].shape, data["past_values"].shape)
        if np.isnan(data["past_values"]).any():
            print("NaNs found in past_values:")
            print(data["past_values"])
            break
        if np.isnan(data["future_values"]).any():
            print("NaNs found in future_values:")
            print(data["future_values"])
            break


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_patchtst_dataset(cfg)


if __name__ == "__main__":
    main()
