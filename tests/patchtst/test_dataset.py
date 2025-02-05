import os
import random
from functools import partial
from pathlib import Path

import hydra
import numpy as np
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, set_seed

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTST
from dystformer.utils import (
    has_enough_observations,
)


def plot_distribution(num_batches: int, batch_size: int, dataset, cfg):
    """
    Check the distribution of dimensions in the dataset across batches.
    """
    # Define training args
    training_args = TrainingArguments(
        run_name=cfg.run_name,
        per_device_train_batch_size=batch_size,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        optim=cfg.train.optim,
        output_dir="outputs",
        logging_strategy="steps",
        logging_steps=cfg.train.log_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        max_steps=cfg.train.max_steps,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        dataloader_num_workers=1,
        torch_compile=False,
        remove_unused_columns=cfg.train.remove_unused_columns,
    )

    model = PatchTST(dict(cfg.patchtst), mode="pretrain")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    dataloader = trainer.get_train_dataloader()

    outputs = model(**next(iter(dataloader)))
    print(outputs)

    for batch in dataloader:
        print(batch)
        break


def test_train_dataset(cfg):
    train_data_dir = os.path.expandvars(cfg.eval.data_path)
    train_data_paths = list(
        filter(lambda file: file.is_file(), Path(train_data_dir).rglob("*"))
    )

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

    plot_distribution(
        num_batches=1000,
        batch_size=cfg.train.per_device_train_batch_size,
        dataset=dataset,
        cfg=cfg,
    )


def get_dim_from_dataset(dataset: FileDataset) -> int:
    """
    Helper function to get system dimension from file dataset
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
    # Convert arrays to structured arrays for efficient row comparison
    dtype = np.dtype((np.void, arr.shape[1] * arr.dtype.itemsize))
    arr_view = np.ascontiguousarray(arr).view(dtype)
    subarr_view = np.ascontiguousarray(subarr).view(dtype)

    # Use numpy's in1d to check if each row in subarr exists in arr
    return np.all(np.in1d(subarr_view, arr_view))


def test_eval_dataset(cfg):
    # get test data paths
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

    system_passed = []
    for system_name, systems in tqdm(
        test_data_dict.items(), total=cfg.eval.num_systems
    ):
        dataset = PatchTSTDataset(
            datasets=systems,
            probabilities=[1.0 / len(systems)] * len(systems),
            context_length=cfg.patchtst.context_length,
            prediction_length=cfg.eval.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            mode="test",
        )

        trajs = [next(iter(sys))["target"].T for sys in systems]
        past_windows, future_windows = list(
            zip(*[(d["past_values"], d["future_values"]) for d in dataset])
        )

        if cfg.eval.window_style == "sampled":
            nw = cfg.eval.num_test_instances
        elif cfg.eval.window_style == "rolling":
            nw = (
                len(trajs[0]) - cfg.patchtst.context_length - cfg.eval.prediction_length
            ) // cfg.eval.window_stride + 1
        elif cfg.eval.window_style == "single":
            nw = 1

        past_contains = all(
            all(
                contains_subarray(traj, past_windows[j])
                for j in range(nw * i, nw * (i + 1))
            )
            for i, traj in enumerate(trajs)
        )
        future_contains = all(
            all(
                contains_subarray(traj, future_windows[j])
                for j in range(nw * i, nw * (i + 1))
            )
            for i, traj in enumerate(trajs)
        )

        system_passed.append(past_contains and future_contains)
    breakpoint()

    assert all(system_passed)
    print("dataset iterates windows before systems. Passed.")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set random seed
    set_seed(seed=cfg.train.seed)

    test_eval_dataset(cfg)


if __name__ == "__main__":
    main()
