# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: modify for dysts

import ast
import logging
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Optional

import typer
from typer_config import use_yaml_config
import torch
import transformers
from transformers import Trainer, TrainingArguments
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from gluonts.transform import LastValueImputation

# from torch.nn.parallel import DistributedDataParallel as DDP
# from chronos import ChronosConfig
from chronos_dysts.model import ChronosConfig

from chronos_dysts.utils import (
    is_main_process,
    log_on_main,
    save_training_info,
    get_next_path,
    load_model,
    has_enough_observations,
)
from chronos_dysts.dataset import ChronosDataset

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 2,
    model_id: str = "google/t5-efficient-tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
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

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    training_data_paths = ast.literal_eval(training_data_paths)
    assert isinstance(training_data_paths, list)

    if isinstance(probability, str):
        probability = ast.literal_eval(probability)
    elif probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    assert isinstance(probability, list)

    assert len(training_data_paths) == len(probability)

    if dataloader_num_workers > len(training_data_paths):
        log_on_main(
            f"Setting the number of data loader workers to {len(training_data_paths)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(training_data_paths)

    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    assert isinstance(tokenizer_kwargs, dict)

    assert model_type in ["seq2seq", "causal"]

    output_dir = get_next_path("run", base_dir=output_dir, file_type="")

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(training_data_paths)} datasets "
        f"for training: {training_data_paths}",
        logger,
    )

    log_on_main(
        f"Mixing probabilities: {probability}",
        logger,
    )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    log_on_main("Initializing model", logger)

    model = load_model(
        model_id=model_id,
        model_type=model_type,
        vocab_size=n_tokens,
        random_init=random_init,
        tie_embeddings=tie_embeddings,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
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

    # Add extra items to model config so that it's saved in the ckpt
    model.config.chronos_config = chronos_config.__dict__

    # # This actually makes things slower, but doesnt throw errors at least
    # # torch distributed training, for torchrun (Elastic Launch)
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)    
    # # Initialize the distributed environment with Gloo backend
    # dist.init_process_group(backend='gloo')
    # # move it to the ROCm device
    # model = model.to(torch.device(f'cuda:{local_rank}'))
    # # Wrap model with DistributedDataParallel
    # model = DDP(model, device_ids=[local_rank])

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

    # Define training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
    )
    log_on_main("Training", logger)

    trainer.train()

    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final", training_config=raw_training_config
        )
    

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()
