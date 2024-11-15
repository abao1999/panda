import os
from functools import partial
from pathlib import Path

import hydra
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from transformers import Trainer, TrainingArguments, set_seed

from dystformer.augmentations import FixedDimensionDelayEmbeddingTransform
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

    model = PatchTST(dict(cfg.patchtst), mode="predict")
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


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set random seed
    set_seed(seed=cfg.train.seed)

    train_data_dir = os.path.expandvars("$WORK/data/big_flow_skew_systems/")
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
        transforms=[
            FixedDimensionDelayEmbeddingTransform(15),
        ],
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    plot_distribution(
        num_batches=1000,
        batch_size=cfg.train.per_device_train_batch_size,
        dataset=dataset,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
