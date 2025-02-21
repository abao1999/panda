import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dystformer.augmentations import (
    RandomAffineTransform,
    RandomConvexCombinationTransform,
    RandomDimSelectionTransform,
)
from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.patchtst import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    PatchTSTForPretraining,
)
from dystformer.utils import (
    has_enough_observations,
    log_on_main,
)

logger = logging.getLogger(__name__)


def kurtosis(
    x: torch.Tensor, dim: int | tuple[int, ...] = -1, fisher: bool = True
) -> torch.Tensor:
    mean = x.mean(dim=dim)
    std = x.std(dim=dim)
    fourth_mom = (x - mean).pow(4).mean(dim=dim)
    kurt = fourth_mom / std.pow(4)
    if fisher:
        kurt -= 3
    return kurt


def load_model(
    mode: str,
    model_config: dict[str, Any],
    pretrained_encoder_path: str | None = None,
    device: str | torch.device | None = None,
) -> PatchTSTForPretraining | PatchTSTForPrediction:
    """
    Load a PatchTST model in either pretraining or prediction mode.

    Args:
        mode: Either "pretrain" or "predict" to specify model type
        model_config: Dictionary containing model configuration parameters
        pretrained_encoder_path: Optional path to pretrained encoder weights for prediction mode

    Returns:
        PatchTSTForPretraining or PatchTSTForPrediction model instance
    """
    config = PatchTSTConfig(**model_config)
    if mode == "pretrain":
        model = PatchTSTForPretraining(config)
    elif mode == "predict":
        model = PatchTSTForPrediction(config)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if pretrained_encoder_path is not None and mode == "predict":
        logger.info(f"Loading pretrained encoder from {pretrained_encoder_path}")
        pretrained_model = PatchTSTForPretraining.from_pretrained(
            pretrained_encoder_path
        )

        # Replace the current encoder with the pretrained encoder
        if hasattr(pretrained_model, "model"):
            pretained_trunk = getattr(pretrained_model, "model")
            assert hasattr(pretained_trunk, "encoder"), "PatchTST must have an encoder"
            model.model.encoder = pretained_trunk.encoder
        else:
            raise Exception("No model found in pretrained model")

    return model.to(device)  # type: ignore


def load_datasets(cfg) -> DataLoader:
    # get train data paths
    train_data_dir_lst = cfg.train_data_dirs
    train_data_paths = []
    for train_data_dir in train_data_dir_lst:
        train_data_dir = os.path.expandvars(train_data_dir)
        train_data_paths.extend(
            filter(lambda file: file.is_file(), Path(train_data_dir).rglob("*"))
        )

    log_on_main(
        f"Loading and filtering {len(train_data_paths)} datasets for training from directories: {train_data_dir_lst}",
        logger,
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

    # adapt number of workers to the number of datasets if there are more workers than datasets
    dataloader_num_workers = cfg.train.dataloader_num_workers
    if dataloader_num_workers > len(train_datasets):
        log_on_main(
            f"Setting the number of data loader workers to {len(train_datasets)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(train_datasets)

    augmentations = [
        RandomConvexCombinationTransform(num_combinations=10, alpha=1.0),
        RandomAffineTransform(out_dim=6, scale=1.0),
    ]
    log_on_main(f"Using augmentations: {augmentations}", logger)

    transforms: list = [RandomDimSelectionTransform(num_dims=cfg.fixed_dim)]

    dataset = PatchTSTDataset(
        datasets=train_datasets,
        probabilities=probability,
        context_length=cfg.patchtst.context_length,
        prediction_length=cfg.patchtst.prediction_length,
        mode="train",
        model_type=cfg.patchtst.mode,
        augmentations=augmentations,
        augmentation_rate=cfg.augmentations.augmentation_rate,
        transforms=transforms,
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    return DataLoader(
        dataset,
        batch_size=cfg.train.per_device_train_batch_size,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )


def dataset_range(dataset: DataLoader, num_samples: int = 1000):
    context_min, context_max = float("inf"), float("-inf")
    future_min, future_max = float("inf"), float("-inf")
    for i, batch in tqdm(zip(range(num_samples), dataset), total=num_samples):
        past_values, future_values = batch["past_values"], batch["future_values"]
        context_min = min(context_min, past_values.min().item())
        context_max = max(context_max, past_values.max().item())
        future_min = min(future_min, future_values.min().item())
        future_max = max(future_max, future_values.max().item())

    return context_min, context_max, future_min, future_max


@torch.no_grad()
def analyze_distribution(
    model: PatchTSTForPrediction | PatchTSTForPretraining,
    dataset: DataLoader,
    num_samples: int = 1000,
    save_path: str = "figures/distribution.png",
):
    predictions, controls, labels = [], [], []
    pred_losses, control_losses = [], []
    model.eval()
    for i, batch in tqdm(zip(range(num_samples), dataset), total=num_samples):
        past_values, future_values = (
            batch["past_values"].to(model.device),
            batch["future_values"].to(model.device),
        )
        loss_val, horizon, preds, loc, scale, *rest = model(
            past_values=past_values,
            future_values=future_values,
            return_dict=False,
            schedule_param=15,
        )

        # horizon = (horizon - loc) / scale
        # preds = (preds - loc) / scale

        label_mean = horizon.mean(dim=1, keepdim=True)
        label_std = horizon.std(dim=1, keepdim=True)
        control = label_mean + label_std * torch.randn_like(horizon)

        pred_losses.append(loss_val.detach().cpu().item())
        control_losses.append(model.loss(control, horizon).detach().cpu().item())

        predictions.append(preds.detach().cpu())
        controls.append(control.detach().cpu())
        labels.append(horizon.detach().cpu())

    predictions = torch.cat(predictions, dim=0)
    controls = torch.cat(controls, dim=0)
    labels = torch.cat(labels, dim=0)

    pred_loss = np.mean(pred_losses)
    control_loss = np.mean(control_losses)

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for ax, data, title in zip(
        axes, [predictions, labels, controls], ["predictions", "labels", "controls"]
    ):
        label = {
            "predictions": f"Prediction Loss: {pred_loss:.2f}",
            "controls": f"Control Loss: {control_loss:.2f}",
            "labels": "Ground Truth",
        }[title]
        kurt = kurtosis(data, dim=(0, 1), fisher=True).mean().item()
        ax.hist(data.numpy().flatten(), bins=1000, log=True, label=label)
        ax.legend()
        ax.set_title(f"{title} (kurtosis: {kurt:.2f})")
        ax.set_xlabel("Value")

    fig.savefig(save_path)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set random seed
    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

    model = load_model(
        mode="predict",
        model_config=cfg.patchtst,
        pretrained_encoder_path=cfg.patchtst.pretrained_encoder_path,
        device="cuda:0",
    )

    dataset = load_datasets(cfg)

    # context_min, context_max, future_min, future_max = dataset_range(dataset)
    # print(f"Context range: {context_min} to {context_max}")
    # print(f"Future range: {future_min} to {future_max}")

    save_dir = "figures/distribution"
    os.makedirs(save_dir, exist_ok=True)

    analyze_distribution(
        model,
        dataset,
        num_samples=1000,
        save_path=os.path.join(save_dir, "distribution.png"),
    )


if __name__ == "__main__":
    main()
