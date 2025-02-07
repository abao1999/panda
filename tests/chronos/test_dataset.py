import logging
import os
import random
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from gluonts.transform import LastValueImputation

from dystformer.augmentations import (
    RandomAffineTransform,
    RandomConvexCombinationTransform,
)
from dystformer.chronos.dataset import ChronosDataset
from dystformer.chronos.tokenizer import ChronosConfig
from dystformer.utils import (
    has_enough_observations,
    log_on_main,
)


def test_train_dataset(cfg):
    # get tokenizer kwargs dict
    tokenizer_kwargs = dict(cfg.chronos.tokenizer_kwargs)

    # check model type is valid
    assert cfg.chronos.model_type in ["seq2seq", "causal"]

    # Get list of files to use for training
    # add all files in train_data_dir to train_data_paths, a list of arrow data filepaths
    train_data_paths = []
    if cfg.train_data_dirs is not None:
        for train_data_dir in cfg.train_data_dirs:
            train_data_paths.extend(
                list(
                    filter(lambda file: file.is_file(), Path(train_data_dir).rglob("*"))
                )
            )

    log_on_main(f"Loading and filtering {len(train_data_paths)} datasets ", logger)

    # load datasets and apply loading filters on the fly
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=cfg.min_past + cfg.chronos.prediction_length,
                max_missing_prop=cfg.max_missing_prop,
            ),
            FileDataset(path=Path(data_path), one_dim_target=False, freq="h"),  # type: ignore
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

    # adapt number of workers to the number of datasets if there are more workers than datasets
    dataloader_num_workers = cfg.train.dataloader_num_workers
    if dataloader_num_workers > len(train_datasets):
        log_on_main(
            f"Setting the number of data loader workers to {len(train_datasets)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(train_datasets)

    # Note: these augmentations are applied to the multivariate target tra
    augmentations = [
        RandomConvexCombinationTransform(num_combinations=10, alpha=1.0),
        RandomAffineTransform(out_dim=6, scale=1.0),
    ]

    chronos_config = ChronosConfig(
        tokenizer_class=cfg.chronos.tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=cfg.chronos.n_tokens,
        n_special_tokens=cfg.chronos.n_special_tokens,
        pad_token_id=cfg.chronos.pad_token_id,
        eos_token_id=cfg.chronos.eos_token_id,
        use_eos_token=cfg.chronos.use_eos_token,
        model_type=cfg.chronos.model_type,
        context_length=cfg.chronos.context_length,
        prediction_length=cfg.chronos.prediction_length,
        num_samples=cfg.chronos.num_samples,
        temperature=cfg.chronos.temperature,
        top_k=cfg.chronos.top_k,
        top_p=cfg.chronos.top_p,
    )

    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=chronos_config.create_tokenizer(),
        context_length=cfg.chronos.context_length,
        prediction_length=cfg.chronos.prediction_length,
        min_past=cfg.min_past,
        model_type=cfg.chronos.model_type,
        imputation_method=LastValueImputation()
        if cfg.chronos.model_type == "causal"
        else None,
        mode="train",
        augmentations=augmentations,
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    for i, data in zip(range(100), shuffled_train_dataset):
        print(f"{i=}")
        for key, value in data.items():
            print(key, value.shape)


def get_dim_from_dataset(dataset: FileDataset) -> int:
    """
    helper function to get system dimension from file dataset
    """
    return next(iter(dataset))["target"].shape[0]


def contains_subarray(arr: np.ndarray, subarr: np.ndarray) -> bool:
    """Check if all rows in subarr exist in arr.

    Args:
        arr: Array of shape (N, D)
        subarr: Array of shape (M, D) where M < N

    Returns:
        True if all rows in subarr exist in arr
    """
    return np.all(np.in1d(subarr.flatten(), arr.flatten()))


def test_eval_dataset(cfg):
    all_ins = []
    # get test data paths, collect all time series for each system in a dict (system_name -> list of FileDatasets)
    test_data_dir = os.path.expandvars(cfg.eval.data_path)
    test_data_dict = {}
    system_dirs = [d for d in Path(test_data_dir).iterdir() if d.is_dir()]
    for system_dir in random.sample(system_dirs, cfg.eval.num_systems):
        system_name = system_dir.name
        system_files = list(system_dir.glob("*"))
        test_data_dict[system_name] = [
            FileDataset(path=Path(file_path), freq="h", one_dim_target=False)
            for file_path in system_files
            if file_path.is_file()
        ]
    for system_name, systems in test_data_dict.items():
        dim = get_dim_from_dataset(systems[0])

        ds = ChronosDataset(
            datasets=systems,
            probabilities=[1.0 / len(systems)] * len(systems),
            tokenizer=None,
            context_length=cfg.chronos.context_length,
            prediction_length=cfg.eval.prediction_length,  # NOTE: should match the forecast prediction length
            min_past=cfg.min_past,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            model_type=cfg.chronos.model_type,
            imputation_method=LastValueImputation()
            if cfg.chronos.model_type == "causal"
            else None,
            mode="test",
        )
        trajs = [next(iter(sys))["target"].T for sys in systems]
        past_windows, future_windows = list(
            zip(*[(d["past_values"], d["future_values"]) for d in ds])
        )
        if cfg.eval.window_style == "sampled":
            nw = cfg.eval.num_test_instances
        elif cfg.eval.window_style == "rolling":
            nw = (
                len(trajs[0]) - cfg.chronos.context_length - cfg.eval.prediction_length
            ) // cfg.eval.window_stride + 1
        elif cfg.eval.window_style == "single":
            nw = 1

        past_arr = (
            np.array(past_windows)
            .squeeze()
            .reshape(len(systems), dim, nw, -1)
            .transpose(0, 2, 3, 1)
        )
        future_arr = (
            np.array(future_windows)
            .squeeze()
            .reshape(len(systems), dim, nw, -1)
            .transpose(0, 2, 3, 1)
        )

        past_all_in = all(
            all(contains_subarray(traj, past_arr[i, j, :]) for j in range(nw))
            for i, traj in enumerate(trajs)
        )
        future_all_in = all(
            all(contains_subarray(traj, future_arr[i, j, :]) for j in range(nw))
            for i, traj in enumerate(trajs)
        )
        all_ins.append(past_all_in and future_all_in)

    assert all(all_ins)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set random seed
    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

    test_eval_dataset(cfg)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
