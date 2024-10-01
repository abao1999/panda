import logging
import os
from functools import partial
from pathlib import Path

import hydra
import torch
import transformers
import wandb
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

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


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set up wandb project and logging if enabled
    if cfg.wandb.log:
        run = wandb.init(
            project=cfg.wandb.project_name,
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

    # Get the path for "$WORK/data/train/Lorenz"
    train_data_dir = os.path.expandvars("$WORK/data/train/Lorenz")
    train_data_paths = [os.path.join(train_data_dir, "5_T-1024.arrow")]
    # if cfg.train_data_dir is not None:
    #     train_data_paths = list(
    #         filter(lambda file: file.is_file(), Path(cfg.train_data_dir).rglob("*"))
    #     )

    # create a new output directory to save results
    output_dir = get_next_path("run", base_dir=Path(cfg.train.output_dir), file_type="")
    print("output_dir: ", output_dir)

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(train_data_paths)} datasets "
        f"for training: {train_data_paths}",
        logger,
    )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=cfg.min_past + cfg.prediction_length,
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

    log_on_main(
        f"Mixing probabilities: {probability}",
        logger,
    )

    log_on_main("Initializing model", logger)

    shuffled_train_dataset = PatchTSTDataset(
        datasets=train_datasets,
        probabilities=probability,
        context_length=512,
        prediction_length=64,
        mode="train",
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    model = PatchTSTModel(dict(cfg.patchtst), mode="pretrain")

    # Define training args
    training_args = TrainingArguments(
        run_name=cfg.run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
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
        remove_unused_columns=cfg.train.remove_unused_columns,
    )

    # check if model weights are contiguous in memory; if not, make them contiguous tensors.
    # This speeds up training and allows checkpoint saving by transformers Trainer
    ensure_contiguous(model)

    # Create Trainer instance and start training
    # TODO: utilize custom callbacks https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/callback#transformers.integrations.WandbCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
        callbacks=[],  # if not cfg.wandb.log else [WandbCallback()], # this duplicates our current logging. Try using custom callback instead
    )

    log_on_main("Training", logger)
    trainer.train()  # Transformers trainer will save model checkpoints automatically

    # terminate wandb run after training
    if cfg.wandb.log:
        # wandb.log(results) # log results to wandb
        run.finish()

    # save final model checkpoint and training info locally
    if is_main_process():
        # ensure_contiguous(model)
        model.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final",
            model_config=vars(
                cfg.patchtst
            ),  # use dataclass asdict for more complex dataclasses
            training_config=OmegaConf.to_container(cfg.train, resolve=True),  # type: ignore
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
