# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.# SPDX-License-Identifier: Apache-2.0
import ast
import logging
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Optional

import typer
import numpy as np
from typer_config import use_yaml_config
import torch
import transformers
from gluonts.dataset.common import FileDataset, ListDataset
from gluonts.itertools import Filter, Map
from gluonts.transform import LastValueImputation

from chronos_dysts.tokenizer import ChronosConfig
from chronos_dysts.utils import (
    log_on_main,
    get_next_path,
    has_enough_observations,
)
from chronos_dysts.dataset import ChronosDataset
from chronos_dysts.augmentations import (
    RandomAffineTransform, 
    RandomConvexCombinationTransform,
    RandomProjectedSkewTransform
)


def get_deep_size(obj, seen=None):
    """Recursively finds size of objects, including referenced objects."""
    import sys
    from collections import deque
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Avoid infinite recursion for self-referencing objects
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([get_deep_size(v, seen) for v in obj.values()])
        size += sum([get_deep_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(vars(obj), seen)
    elif isinstance(obj, (list, tuple, set, frozenset, deque)):
        size += sum([get_deep_size(i, seen) for i in obj])
    
    return size

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
@use_yaml_config(param_name="config")
def main(
    train_data_dir: str,
    extra_train_data_paths: Optional[str] = None,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    shuffle_buffer_length: int = 100,
    model_type: str = "seq2seq",
    output_dir: str = "./output/",
    tf32: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
):
    if tf32 and not (
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
        tf32 = False

    if seed is None:
        seed = random.randint(0, 2**32)

    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)

    output_dir = Path(output_dir)

    # add all files in train_data_dir to train_data_paths, a list of arrow data filepaths
    train_data_paths = []
    if train_data_dir is not None:
        train_data_paths = list(
            filter(lambda file: file.is_file(), Path(train_data_dir).rglob('*'))
        )
    print("train data paths: ", train_data_paths)

    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    assert isinstance(tokenizer_kwargs, dict)

    assert model_type in ["seq2seq", "causal"]

    output_dir = get_next_path("run", base_dir=output_dir, file_type="")

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(train_data_paths)} datasets "
        f"for training: {train_data_paths}",
        logger,
    )

    def make_coord_iterator(dataset):
        yield from next(iter(dataset))["target"]

    class CoordIterator:

        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            return self._Generator(self.dataset)

        class _Generator:

            def __init__(self, dataset):
                self.data = next(iter(dataset))

            def __iter__(self):

                yield {}

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=data_path, freq='h', one_dim_target=False)
        )
        for data_path in train_data_paths
    ]


    train_datasets = [
        make_coord_iterator(dataset)
        for dataset in train_datasets
    ]
    print(train_datasets[0])
    print(get_deep_size(train_datasets[0]))
    print(get_deep_size([i for i in train_datasets[0]]))
    for i in train_datasets[0]:
        print(i)

    fsdfsadfsdf

    # train_datasets=[
    #     RandomConvexCombinationTransform(10, 1, 999)(dataset)
    #     for dataset in train_datasets[:len(train_data_paths)]
    # ]
    
    # set probabilities (how we weight draws from each data file)
    if isinstance(probability, str):
        probability = ast.literal_eval(probability)
    elif probability is None:
        probability = [1.0 / len(train_datasets)] * len(train_datasets)
    assert isinstance(probability, list)

    assert len(train_datasets) == len(probability)

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

    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=chronos_config.create_tokenizer(),
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_type=model_type,
        imputation_method=LastValueImputation() if model_type == "causal" else None,
        mode="training",
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    
    for i, x in enumerate(shuffled_train_dataset):
        # print(i, x)
        pass
    

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()
