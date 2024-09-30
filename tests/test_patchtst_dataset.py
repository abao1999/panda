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
    train_data_dir = os.path.expandvars("$WORK/data/train/Lorenz")
    train_data_paths = [os.path.join(train_data_dir, "3_T-1024.arrow")]
    # if cfg.train_data_dir is not None:
    #     train_data_paths = list(
    #         filter(lambda file: file.is_file(), Path(cfg.train_data_dir).rglob("*"))
    #     )

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
        context_length=512,
        prediction_length=64,
        mode="train",
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    for data in dataset:
        print(data)
        print(data["future_values"].shape, data["past_values"].shape)
        print(any(np.isnan(data["past_values"])))
        print(any(np.isnan(data["future_values"])))

    # chronos_config = ChronosConfig(
    #     tokenizer_class=cfg.tokenizer_class,
    #     tokenizer_kwargs=dict(cfg.tokenizer_kwargs),
    #     n_tokens=cfg.n_tokens,
    #     n_special_tokens=cfg.n_special_tokens,
    #     pad_token_id=cfg.pad_token_id,
    #     eos_token_id=cfg.eos_token_id,
    #     use_eos_token=cfg.use_eos_token,
    #     model_type=cfg.model_type,
    #     context_length=cfg.context_length,
    #     prediction_length=cfg.prediction_length,
    #     num_samples=cfg.num_samples,
    #     temperature=cfg.temperature,
    #     top_k=cfg.top_k,
    #     top_p=cfg.top_p,
    # )

    # shuffled_train_dataset = ChronosDataset(
    #     datasets=train_datasets,
    #     probabilities=probability,
    #     tokenizer=chronos_config.create_tokenizer(),
    #     context_length=cfg.context_length,
    #     prediction_length=cfg.prediction_length,
    #     min_past=cfg.min_past,
    #     model_type=cfg.model_type,
    #     mode="train",
    # ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    # for data in shuffled_train_dataset:
    #     print(data)
    #     break


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_patchtst_dataset(cfg)


if __name__ == "__main__":
    main()
