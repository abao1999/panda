import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from transformers import Trainer, TrainingArguments

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTSTModel
from dystformer.utils import (
    has_enough_observations,
)


def fixed_dim_collator(
    features: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collates trajectories by sampling a fixed dimension and filtering out any
    trajectories that do not match the sampled dimension.
    Args:
        features: list of dictionaries with "past_values" and "future_values" tensors
            with shapes [num_dimensions, context_length] and [num_dimensions, prediction_length].
            The input list has length batch_size * num_visible_devices.

    Returns:
        Dict with "past_values" and "future_values" flattened and equalized in dimension 0,
            with size equalized_dim*batch_size
    """

    grouped_features = {}
    for feature in features:
        key = feature["past_values"].shape[-1]
        if key not in grouped_features:
            grouped_features[key] = []
        grouped_features[key].append(feature)

    sampled_dim = random.choice(list(grouped_features.keys()))
    batch = {
        key: torch.stack([f[key] for f in grouped_features[sampled_dim]])
        for key in grouped_features[sampled_dim][0].keys()
    }
    return batch


def dimensional_collator(
    features: List[Dict[str, torch.Tensor]],
    equalized_dim: int,
) -> Dict[str, torch.Tensor]:
    """
    Collates trajectories into flat data by stacking along dimension

    Args:
        features: list of dictionaries with "past_values" and "future_values" tensors
            with shapes [num_dimensions, context_length] and [num_dimensions, prediction_length].
            The input list has length batch_size * num_visible_devices.

    Returns:
        Dict with "past_values" and "future_values" flattened and equalized in dimension 0,
            with size equalized_dim*batch_size
    """
    dims = np.asarray([feature["past_values"].shape[-1] for feature in features])
    batch_size = len(features)
    min_dim = min(dims)

    # ensure that the equalization dimension satisfies the constraint:
    # k * min(dims) <= k * equalized_dim <= sum(dims) <= k * max(dims)
    # NOTE: this will still cause non-uniform batch sizes,
    #  but only in multiples of the number of devices
    while not (batch_size * min_dim <= batch_size * equalized_dim <= sum(dims)):
        if equalized_dim < min_dim:
            equalized_dim = min_dim
        else:  # decrement until the constraint is satisfied
            equalized_dim -= 1

    # randomly choose dimensions to remove in order to equalize across the batch
    equalization_budget = sum(dims) - batch_size * equalized_dim

    if equalization_budget > 0:
        cuts = np.random.choice(equalization_budget, size=len(features) - 1)
        num_remove = np.diff(
            np.concatenate([[0], np.sort(cuts), [equalization_budget]])
        )

        # resample if the constraints are not satisfied
        while np.any(num_remove > dims):
            cuts = np.random.choice(equalization_budget, size=len(features) - 1)
            num_remove = np.diff(
                np.concatenate([[0], np.sort(cuts), [equalization_budget]])
            )
    else:
        num_remove = np.zeros(len(features))

    num_keep = (dims - num_remove).astype(int)
    batch = {
        k: torch.concat(
            [
                feat[k][
                    :, np.random.choice(feat[k].shape[1], size=nkeep, replace=False)
                ]
                for nkeep, feat in zip(num_keep, features)
            ],
            dim=1,
        )
        for k in features[0].keys()
    }
    batch["dims"] = torch.from_numpy(num_keep).to(torch.int32)

    return batch


def check_dim_distribution(num_batches: int, batch_size: int, dataset, cfg):
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
        output_dir="test",
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

    model = PatchTSTModel(dict(cfg.patchtst), mode="pretrain")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=partial(dimensional_collator, equalized_dim=6),
        # data_collator=fixed_dim_collator,
        callbacks=[],  # if not cfg.wandb.log else [WandbCallback()],
    )
    dataloader = trainer.get_train_dataloader()

    # test model forward with custom batch
    batch = next(iter(dataloader))
    output = model(**batch)
    print(output)

    dims = np.concatenate(
        [
            batch["dims"].cpu().numpy()
            for _, batch in zip(range(num_batches), dataloader)
        ]
    )
    print(dims)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels, counts = np.unique(dims, return_counts=True)
    ax.bar(labels, counts, align="center")
    ax.set_title("Dimension Distribution")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Count")
    plt.savefig("figures/naive_dim_distribution.png")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    train_data_dir = os.path.expandvars("$WORK/data/train/")
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

    check_dim_distribution(num_batches=1000, batch_size=16, dataset=dataset, cfg=cfg)


if __name__ == "__main__":
    main()
