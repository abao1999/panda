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
from transformers import Trainer, TrainingArguments, set_seed

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

    model = PatchTSTModel(dict(cfg.patchtst), mode="predict")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    dataloader = trainer.get_train_dataloader()

    loss_val, past_values, patch_input, future_values, preds, loc, scale, *rest = model(
        return_dict=False, **next(iter(dataloader))
    )
    print(past_values.shape)
    print(patch_input.shape)
    print(future_values.shape)
    print(preds.shape)
    print(loc.shape)
    print(scale.shape)
    print(loss_val.shape)

    # Plot trajectories from patch_input
    fig = plt.figure(figsize=(20, 12))

    # 3D plot
    ax_3d = fig.add_subplot(221, projection="3d")
    cmap = plt.get_cmap("Set1")
    colors = [cmap(i % cmap.N) for i in range(batch_size)]

    # Add a single representative line for each linestyle
    ax_3d.plot([], [], color="black", linestyle="-", label="Predictions")
    ax_3d.plot([], [], color="black", linestyle="-.", label="True")
    ax_3d.plot([], [], color="black", linestyle=":", label="Past")

    # Time series subplots
    ax_x = fig.add_subplot(222)
    ax_y = fig.add_subplot(223)
    ax_z = fig.add_subplot(224)

    for b in range(batch_size):
        loc_b = loc[b].detach().cpu().numpy().T
        scale_b = scale[b].detach().cpu().numpy().T
        past_values_b = past_values[b].detach().cpu().numpy().T
        trajectory = loc_b + scale_b * preds[b].detach().cpu().numpy().T
        true_trajectory = future_values[b].detach().cpu().numpy().T

        # 3D plot
        ax_3d.scatter(
            *loc_b,
            color=colors[b],
            marker="o",
            facecolors="none",
            s=40,
            label="mean" if b == 0 else "",
        )
        ax_3d.plot(*trajectory, color=colors[b], linewidth=0.5, linestyle="-")
        ax_3d.plot(*past_values_b, color=colors[b], linewidth=0.5, linestyle=":")
        ax_3d.plot(*true_trajectory, color=colors[b], linestyle="-.")

        # Time series plots
        time = np.arange(len(past_values_b[0]) + len(trajectory[0]))
        for ax, coord in zip([ax_x, ax_y, ax_z], range(3)):
            ax.plot(
                time[: len(past_values_b[0])],
                past_values_b[coord],
                color=colors[b],
                linestyle=":",
                linewidth=0.5,
            )
            ax.plot(
                time[len(past_values_b[0]) :],
                trajectory[coord],
                color=colors[b],
                linestyle="-",
                linewidth=0.5,
            )
            ax.plot(
                time[len(past_values_b[0]) :],
                true_trajectory[coord],
                color=colors[b],
                linestyle="-.",
                linewidth=0.5,
            )
            ax.set_title(f"Coordinate {coord+1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")

    ax_3d.legend()
    plt.tight_layout()
    plt.savefig("figures/patch_input_trajectories.png")
    plt.close()
    fsdfs
    outputs = zip(
        *[
            model(return_dict=False, **batch)
            for _, batch in zip(range(num_batches), dataloader)
        ]
    )
    future_values, preds, loc, scale, loss_val, *rest = outputs
    print(future_values)
    print(future_values.shape)
    print(preds.shape)
    print(loc.shape)
    print(scale.shape)
    print(loss_val.shape)

    agg_preds = (
        torch.concat([torch.mean(pred, dim=1) for pred in preds], dim=0).detach().cpu()
    )
    agg_future_values = torch.concat(future_values, dim=0).detach().cpu()
    agg_loc = torch.concat(loc, dim=0).detach().cpu().mean(dim=0)
    agg_scale = torch.concat(scale, dim=0).detach().cpu().mean(dim=0)
    agg_loss = [torch.log10(loss.detach().cpu()) for loss in loss_val]
    print(agg_loc.shape)
    print(agg_scale.shape)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(3):
        ax = axes[i]
        ax.hist(torch.log10(agg_preds[:, i]), bins=100, alpha=0.5, label="Predictions")
        ax.hist(
            torch.log10(agg_future_values[:, i]),
            bins=100,
            alpha=0.5,
            label="Future Values",
        )
        # ax.axvline(agg_loc[i], color="red")
        # ax.axvspan(
        #     agg_loc[i].item() - agg_scale[i].item(),
        #     agg_loc[i].item() + agg_scale[i].item(),
        #     alpha=0.3,
        #     color="red",
        # )
        ax.set_title(f"Norms for dim {i}")
        ax.set_xlabel("Log10 Norm")
        ax.set_ylabel("Count")
        ax.legend()
    axes[-1].hist(agg_loss, bins=100)
    axes[-1].set_title("Log Loss")
    plt.savefig("figures/norms_distribution.png")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set random seed
    set_seed(seed=cfg.train.seed)

    train_data_dir = os.path.expandvars("$WORK/data/nonstandardized_train/")
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
        fixed_dim=cfg.fixed_dim,
        delay_embed_prob=cfg.delay_embed_prob,
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    plot_distribution(
        num_batches=1000,
        batch_size=cfg.train.per_device_train_batch_size,
        dataset=dataset,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
