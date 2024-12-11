"""
Training/fine-tuning script, adapted from chronos-forecasting
"""

import logging
from functools import partial
from pathlib import Path

import hydra
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from gluonts.transform import LastValueImputation
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

import dystformer.augmentations as augmentations
import wandb
from dystformer.chronos.dataset import ChronosDataset
from dystformer.chronos.tokenizer import ChronosConfig
from dystformer.utils import (
    ensure_contiguous,
    get_next_path,
    has_enough_observations,
    is_main_process,
    load_model,
    log_on_main,
    sample_index_pairs,
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

    # get tokenizer kwargs dict
    tokenizer_kwargs = dict(cfg.chronos.tokenizer_kwargs)

    # check model type is valid
    assert cfg.chronos.model_type in ["seq2seq", "causal"]

    # Get list of files to use for training
    # add all files in train_data_dirs to train_data_paths, a list of arrow data filepaths
    train_data_paths = []
    if cfg.train_data_dirs is not None:
        for train_data_dirs in cfg.train_data_dirs:
            train_data_paths.extend(
                list(
                    filter(
                        lambda file: file.is_file(), Path(train_data_dirs).rglob("*")
                    )
                )
            )
    log_on_main(f"train_data_paths: {train_data_paths}", logger)

    # add any additional arrow data filepaths specified to our training set
    if cfg.extra_train_data_paths is not None:
        extra_paths = [
            Path(file) for file in cfg.extra_train_data_paths if Path(file).is_file()
        ]
        assert isinstance(extra_paths, list), "extra paths must be a list literal"
        train_data_paths.extend(extra_paths)

    # create a new output directory to save results
    output_dir = get_next_path("run", base_dir=Path(cfg.train.output_dir), file_type="")
    print("output_dir: ", output_dir)

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(train_data_paths)} datasets "
        f"for training: {train_data_paths}",
        logger,
    )

    # load datasets and apply loading filters on the fly
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=cfg.min_past + cfg.chronos.prediction_length,
                max_missing_prop=cfg.max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),  # type: ignore
        )
        for data_path in train_data_paths
    ]

    # apply augmentations on the fly
    # TODO: understand the fine-tuning details more. Do we want to aggregate the samples together, on the fly?
    # TODO: also will probably need to re-weight probabilities to take into account the type of augmentation
    #    - say original data is (3,1024) and augmented is (10,1024). Then each entry in the augmented would have less probability under current scheme
    #    - essentially, the training datasets is jagged arrays of different lengths
    #    - (also, if we're doing a skew transform, we might want to weight the original data more heavily?)

    # system-scale augmentations
    log_on_main("Applying system-scale augmentations", logger)
    for augmentation_cls_name in cfg.augmentations.system:
        augmentation_cls = getattr(augmentations, augmentation_cls_name)
        log_on_main(
            f"Applying {augmentation_cls.__name__} system-scale augmentation", logger
        )
        kwargs = dict(getattr(cfg.augmentations, f"{augmentation_cls_name}_kwargs"))
        augmentation_fn = partial(augmentation_cls, **kwargs)
        train_datasets.extend(
            [augmentation_fn(ds) for ds in train_datasets[: len(train_data_paths)]]
        )

    # ensemble-scale augmentations
    log_on_main("Applying ensemble-scale augmentations", logger)
    for augmentation_cls_name in cfg.augmentations.ensemble:
        augmentation_cls = getattr(augmentations, augmentation_cls_name)
        log_on_main(
            f"Applying {augmentation_cls.__name__} ensemble-scale augmentation", logger
        )
        kwargs = dict(getattr(cfg.augmentations, f"{augmentation_cls_name}_kwargs"))
        augmentation_fn = partial(augmentation_cls, **kwargs)
        train_datasets.extend(
            [
                augmentation_fn(train_datasets[i], train_datasets[j])
                for i, j in sample_index_pairs(len(train_data_paths), num_pairs=5)
            ]
        )

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

    model = load_model(
        model_id=cfg.chronos.model_id,
        model_type=cfg.chronos.model_type,
        vocab_size=cfg.chronos.n_tokens,
        random_init=cfg.chronos.random_init,
        tie_embeddings=cfg.chronos.tie_embeddings,
        pad_token_id=cfg.chronos.pad_token_id,
        eos_token_id=cfg.chronos.eos_token_id,
    )

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

    # Add extra items to model config so that it's saved in the ckpt
    model.config.chronos_config = chronos_config.__dict__

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
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

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
                chronos_config
            ),  # use dataclass asdict for more complex dataclasses
            train_config=OmegaConf.to_container(cfg.train, resolve=True),  # type: ignore
            all_config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
