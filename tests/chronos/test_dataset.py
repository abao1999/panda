import logging
from functools import partial
from pathlib import Path

import hydra
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


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set random seed
    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

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
        print()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
