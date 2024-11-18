import logging
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import hydra
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from omegaconf import OmegaConf
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import wandb
from dystformer.augmentations import (
    RandomAffineTransform,
    RandomConvexCombinationTransform,
    RandomDimSelectionTransform,
)
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


@dataclass
class NoiseScaleScheduler:
    """
    Noise scale scheduler for the training process. Noise to be applied to the instance-normalized trajectories
    Args:
        schedule_name: schedule for the noise scale. Options are "linear", "exponential", "cosine"
            cosine schedule is inspired by IDDPM and is known to work well for training diffusion models
        start: initial noise scale
        end: final noise scale
        decay_rate: decay rate for the exponential decay schedule
        eps: epsilon for the cosine schedule (for numerical stability)
        epoch_stop: epoch (as a fraction of total epochs) at which to stop the schedule
    """

    schedule_name: str
    start: float
    end: float
    decay_rate: float = 8.0
    eps: float = 0.008
    epoch_stop: float = 1.0

    def __post_init__(self):
        if self.schedule_name not in ["linear", "exponential", "cosine"]:
            raise ValueError("Invalid schedule for noise scale scheduler")
        if self.start < self.end:
            raise ValueError("Start noise scale must be greater than end noise scale")

        if self.epoch_stop > 1.0 or self.epoch_stop < 0.0:
            raise ValueError("Epoch stop must be between 0.0 and 1.0")

        self.decay_rate = torch.tensor(self.decay_rate)  # type: ignore
        self.eps = torch.tensor(self.eps)  # type: ignore

        self.schedule_fn = {
            "linear": lambda t: self.start + (self.end - self.start) * t,
            "cosine": lambda t: torch.cos(
                (t + self.eps) / (1 + self.eps) * torch.pi / 2
            )
            ** 2,
            "exponential": lambda t: self.start * torch.exp(-self.decay_rate * t),
        }[self.schedule_name]

    def __call__(self, epoch: float) -> float | torch.Tensor:
        t = epoch / self.epoch_stop
        noise_scale = self.schedule_fn(t)
        if epoch > self.epoch_stop or noise_scale < self.end:
            noise_scale = self.end
        return noise_scale


class NoiseScaleLoggingCallback(TrainerCallback):
    def __init__(
        self,
        noise_scale_scheduler: NoiseScaleScheduler,
        logger: logging.Logger,
        log_interval: int = 100,
    ):
        self.noise_scale_scheduler = noise_scale_scheduler
        self.logger = logger
        self.log_interval = log_interval
        # self.wandb_run = wandb.run if wandb.run is not None else None

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.log_interval == 0 or state.epoch == 1.0:
            epoch = float(state.epoch)  # type: ignore
            noise_scale = self.noise_scale_scheduler(epoch)
            # log_on_main(f"Logging noise_scale to wandb: {noise_scale}", self.logger)

            # If using wandb or tensorboard, log it there as well
            if args.report_to:
                for report in args.report_to:
                    if report == "wandb":
                        wandb.log({"noise_scale": noise_scale}, step=state.global_step)
                    elif report == "tensorboard":
                        if control.should_log:
                            if not hasattr(self, "log_history"):
                                self.log_history = []
                            self.log_history.append({"noise_scale": noise_scale})


class CustomTrainer(Trainer):
    def __init__(
        self,
        model: PatchTST,
        args: TrainingArguments,
        noise_scale_scheduler: NoiseScaleScheduler,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.noise_scale_scheduler = noise_scale_scheduler

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        epoch = float(self.state.epoch)  # type: ignore
        noise_scale = self.noise_scale_scheduler(epoch)

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
    # transforms = [
    #     FixedDimensionDelayEmbeddingTransform(embedding_dim=cfg.fixed_dim),
    # ]
    transforms = [
        RandomDimSelectionTransform(num_dims=cfg.fixed_dim),
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

    if cfg.noiser.enabled:
        noise_scale_scheduler = NoiseScaleScheduler(
            schedule_name=cfg.noiser.schedule_name,
            start=cfg.noiser.start,
            end=cfg.noiser.end,
            decay_rate=cfg.noiser.decay_rate,
            eps=cfg.noiser.eps,
            epoch_stop=cfg.noiser.epoch_stop,
        )

        logging_callback = NoiseScaleLoggingCallback(
            noise_scale_scheduler=noise_scale_scheduler,
            logger=logger,
            log_interval=cfg.noiser.log_steps,
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=shuffled_train_dataset,
            noise_scale_scheduler=noise_scale_scheduler,
            callbacks=[logging_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=shuffled_train_dataset,
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
