import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from omegaconf import OmegaConf
from transformers import (
    Trainer,
    TrainingArguments,
)

import wandb
from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTSTModel
from dystformer.utils import (
    ensure_contiguous,
    get_next_path,
    has_enough_observations,
    is_main_process,
    log_on_main,
    save_training_info,
)

logger = logging.getLogger(__name__)


def fixed_dim_collator(
    features: List[Dict[str, torch.Tensor]],
    fixed_dim: int,
) -> Dict[str, torch.Tensor]:
    """
    Collates trajectories by randomly sampling a fixed number of dimensions and filtering out any
    trajectories that do not match the sampled dimension.

    NOTE:
    - this doesnt perform any out-of-bounds checks and assumes that you can just sample the dims
    in other words, this collator is brittle, but will work for the continuous systems
    - this does the exact same thing as setting cfg.fixed_dim = 3

    NOTE: will be deprecated
    """
    sampled_dims = [
        torch.randperm(f["past_values"].shape[1])[:fixed_dim] for f in features
    ]
    batch = {
        key: torch.stack([f[key][:, sampled_dims[i]] for i, f in enumerate(features)])
        for key in features[0].keys()
    }
    return batch


def random_dim_collator(
    features: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collates trajectories by sampling a dimension and filtering out any
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


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set up wandb project and logging if enabled
    if cfg.wandb.log:
        run = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            config=dict(cfg),
            sync_tensorboard=False,  # auto-upload tensorboard metrics
            group=cfg.wandb.group_name,
            resume=cfg.wandb.resume,
        )

    # set floating point precision
    use_tf32 = cfg.train.tf32
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        use_tf32 = False

    # set random seed
    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

    # get train data paths
    train_data_dir = os.path.expandvars(cfg.train_data_dir)
    # train_data_paths = [
    #     os.path.join(train_data_dir, "Lorenz/0_T-2048.arrow"),
    #     os.path.join(train_data_dir, "ThomasLabyrinth/0_T-2048.arrow"),
    # ]
    train_data_paths = list(
        filter(lambda file: file.is_file(), Path(train_data_dir).rglob("*"))
    )

    # create a new output directory to save results
    output_dir = get_next_path("run", base_dir=Path(cfg.train.output_dir), file_type="")
    print("output_dir: ", output_dir)

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(train_data_paths)} datasets for training",
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

    # # system-scale augmentations
    # log_on_main("Applying system-scale augmentations", logger)
    # for augmentation_cls_name in cfg.augmentations.system:
    #     augmentation_cls = getattr(augmentations, augmentation_cls_name)
    #     log_on_main(
    #         f"Applying {augmentation_cls.__name__} system-scale augmentation", logger
    #     )
    #     kwargs = dict(getattr(cfg.augmentations, f"{augmentation_cls_name}_kwargs"))
    #     augmentation_fn = partial(augmentation_cls, **kwargs)
    #     train_datasets.extend(
    #         [augmentation_fn(ds) for ds in train_datasets[: len(train_data_paths)]]
    #     )

    # # ensemble-scale augmentations
    # log_on_main("Applying ensemble-scale augmentations", logger)
    # for augmentation_cls_name in cfg.augmentations.ensemble:
    #     augmentation_cls = getattr(augmentations, augmentation_cls_name)
    #     log_on_main(
    #         f"Applying {augmentation_cls.__name__} ensemble-scale augmentation", logger
    #     )
    #     kwargs = dict(getattr(cfg.augmentations, f"{augmentation_cls_name}_kwargs"))
    #     augmentation_fn = partial(augmentation_cls, **kwargs)
    #     train_datasets.extend(
    #         [
    #             augmentation_fn(train_datasets[i], train_datasets[j])
    #             for i, j in sample_index_pairs(len(train_data_paths), num_pairs=5)
    #         ]
    #     )

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

    log_on_main("Initializing model", logger)

    shuffled_train_dataset = PatchTSTDataset(
        datasets=train_datasets,
        probabilities=probability,
        context_length=cfg.patchtst.context_length,
        prediction_length=cfg.patchtst.prediction_length,
        mode="train",
        fixed_dim=cfg.fixed_dim,
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    model = PatchTSTModel(dict(cfg.patchtst), mode=cfg.patchtst.mode)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_on_main(f"Total trainable parameters: {trainable_params:,}", logger)

    # Define training args
    training_args = TrainingArguments(
        run_name=cfg.run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        logging_dir=f"wandb/tbruns/{run.name}_{run.id}/logs"
        if cfg.wandb.log
        else str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=cfg.train.log_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        report_to=["wandb"] if cfg.wandb.log else ["tensorboard"],
        max_steps=cfg.train.max_steps,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=use_tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        torch_compile=cfg.train.torch_compile,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        ddp_backend="nccl",
        remove_unused_columns=cfg.train.remove_unused_columns,
        seed=cfg.train.seed,
    )

    # check if model weights are contiguous in memory; if not, make them contiguous tensors.
    # This speeds up training and allows checkpoint saving by transformers Trainer
    ensure_contiguous(model)

    collation_method = {
        "random_dim": random_dim_collator,
        "fixed_dim": partial(fixed_dim_collator, fixed_dim=3),
        "" "dimensional": partial(dimensional_collator, equalized_dim=6),
    }.get(cfg.collate_method)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
        data_collator=collation_method,
    )

    log_on_main("Training", logger)
    trainer.train()  # Transformers trainer will save model checkpoints automatically

    # terminate wandb run after training
    if cfg.wandb.log:
        run.finish()

    # save final model checkpoint and training info locally
    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")  # type: ignore
        save_training_info(
            output_dir / "checkpoint-final",
            model_config=OmegaConf.to_container(cfg.patchtst, resolve=True),  # type: ignore
            training_config=OmegaConf.to_container(cfg.train, resolve=True),  # type: ignore
        )


if __name__ == "__main__":
    main()
