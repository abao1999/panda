import os
from functools import partial
from pathlib import Path

import hydra
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
    train_data_dir = os.path.expandvars("$WORK/data/train/Lorenz")
    train_data_paths = [os.path.join(train_data_dir, "3_T-1024.arrow")]
    # if cfg.train_data_dir is not None:
    #     train_data_paths = list(
    #         filter(lambda file: file.is_file(), Path(cfg.train_data_dir).rglob("*"))
    #     )

    print(train_data_paths)
    dfsdfs

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=cfg.min_past + cfg.prediction_length,
                max_missing_prop=cfg.max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),  # type: ignore
        )
        for data_path in train_data_paths
    ]

    dataset = PatchTSTDataset(
        datasets=[],
        probabilities=[],
        context_length=512,
        prediction_length=64,
    )

    for data in dataset:
        print(data)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_patchtst_dataset(cfg)


if __name__ == "__main__":
    main()
