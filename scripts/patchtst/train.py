import logging
import os
from functools import partial
from pathlib import Path
from typing import List

import hydra
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
from dystformer.augmentations import (
    FixedDimensionDelayEmbeddingTransform,
    RandomAffineTransform,
    RandomConvexCombinationTransform,
)
from dystformer.patchtst.callbacks import AdaptiveNumBinsCallback
from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTST
from dystformer.utils import (
    ensure_contiguous,
    get_next_path,
    has_enough_observations,
    is_main_process,
    log_on_main,
    save_training_info,
)

logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    valid_state_vars = ["num_bins", "noise_scale"]

    def __init__(
        self,
        model: PatchTST,
        adaptive_state_variable_names: List[str],
        args: TrainingArguments,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.state_vars = adaptive_state_variable_names
        if not all(state_var in self.valid_state_vars for state_var in self.state_vars):
            raise ValueError(
                f"Invalid combination of state variables: {self.state_vars}"
            )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        # global_step = self.state.global_step
        epoch = float(self.state.epoch)  # type: ignore
        decay_rate = torch.tensor(5.0)  # Adjust this value to control the decay speed
        noise_scale = 1 * torch.exp(-decay_rate * epoch)
        log_on_main(f"noise_scale: {noise_scale}", logger)

        outputs = model(**inputs, noise_scale=noise_scale)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later (HF comment)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


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
    train_data_dir_lst = cfg.train_data_dirs
    train_data_paths = []
    for train_data_dir in train_data_dir_lst:
        train_data_dir = os.path.expandvars(train_data_dir)
        train_data_paths.extend(
            filter(lambda file: file.is_file(), Path(train_data_dir).rglob("*"))
        )

    # create a new output directory to save results
    output_dir = get_next_path("run", base_dir=Path(cfg.train.output_dir), file_type="")

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loaded and filtered {len(train_data_paths)} datasets for training from directories: {train_data_dir_lst}",
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

    log_on_main(f"number of train_datasets: {len(train_datasets)}", logger)

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

    augmentations = [
        RandomConvexCombinationTransform(num_combinations=10, alpha=1.0),
        RandomAffineTransform(out_dim=6, scale=1.0),
    ]
    transforms = [
        FixedDimensionDelayEmbeddingTransform(embedding_dim=cfg.fixed_dim),
    ]

    log_on_main(f"Using augmentations: {augmentations}", logger)

    shuffled_train_dataset = PatchTSTDataset(
        datasets=train_datasets,
        probabilities=probability,
        context_length=cfg.patchtst.context_length,
        prediction_length=cfg.patchtst.prediction_length,
        mode="train",
        fixed_dim=cfg.fixed_dim,
        augmentations=augmentations,
        transforms=transforms,
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    if (
        cfg.patchtst.mode == "predict"
        and cfg.patchtst.pretrained_encoder_path is not None
    ):
        log_on_main(
            f"Loading pretrained encoder from {cfg.patchtst.pretrained_encoder_path}",
            logger,
        )

    model = PatchTST(
        dict(cfg.patchtst),
        mode=cfg.patchtst.mode,
        pretrained_encoder_path=cfg.patchtst.pretrained_encoder_path,
    )  # .to("cuda")

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
        dataloader_prefetch_factor=cfg.train.dataloader_prefetch_factor,
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

    adaptive_callbacks = {}
    if cfg.quantizer.enabled:
        adaptive_quantization_callback = AdaptiveNumBinsCallback(
            initial_bins=cfg.quantizer.initial_bins,
            max_bins=cfg.quantizer.max_bins,
            step_interval=cfg.quantizer.step_interval,
            num_bins_growth_factor=cfg.quantizer.num_bins_growth_factor,
            logger=logger,
        )
        adaptive_callbacks["num_bins"] = adaptive_quantization_callback

    if cfg.noiser.enabled:
        adaptive_callbacks["noise_scale"] = (
            None  # callback is very slow, do not mutate trainer state!
        )

    log_on_main(f"Using adaptive callbacks: {adaptive_callbacks}", logger)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
        adaptive_state_variable_names=list(adaptive_callbacks.keys()),
    )
    # else:
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=shuffled_train_dataset,
    #     )

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
