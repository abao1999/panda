import logging
import os
from pathlib import Path

import hydra
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import batcher

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTSTModel
from dystformer.utils import log_on_main

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
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
    test_data_dir = os.path.expandvars("$WORK/data/test/")
    test_data_paths = list(
        filter(lambda file: file.is_file(), Path(test_data_dir).rglob("*"))
    )
    test_datasets = [
        FileDataset(path=Path(data_path), freq="h", one_dim_target=False)
        for data_path in test_data_paths
    ]

    # set probabilities (how we weight draws from each data file)
    if isinstance(cfg.probability, float):
        probability = cfg.probability
    elif cfg.probability is None:
        probability = [1.0 / len(test_datasets)] * len(test_datasets)
    assert isinstance(probability, list)

    assert len(test_datasets) == len(probability)

    test_dataset = PatchTSTDataset(
        datasets=test_datasets,
        probabilities=probability,
        context_length=cfg.patchtst.context_length,
        prediction_length=cfg.patchtst.prediction_length,
        mode="test",
    )

    model = PatchTSTModel.from_pretrained(
        pretrain_path=cfg.eval.checkpoint_path,
        mode="pretrain",
    )
    model.eval()

    for batch in batcher(test_dataset, batch_size=4):
        print(len(batch))


if __name__ == "__main__":
    main()
