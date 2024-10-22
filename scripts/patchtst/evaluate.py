import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import transformers
from dysts.metrics import compute_metrics
from gluonts.dataset.common import FileDataset
from gluonts.itertools import batcher
from tqdm.auto import tqdm

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTST
from dystformer.utils import log_on_main

logger = logging.getLogger(__name__)


def evaluate_forecasting_model(
    model: PatchTST,
    systems: Dict[str, PatchTSTDataset],
    batch_size: int,
    metrics: Optional[List[str]] = None,
    return_predictions: bool = False,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    """
    Evaluate the model on each test system and save metrics.

    Args:
        model: The PatchTST model to evaluate.
        systems: A dictionary mapping system names to their respective PatchTSTDataset.
        batch_size: The batch size to use for evaluation.
        metrics: Optional list of metric names to compute.
        return_predictions: Whether to return the predictions.
    Returns:
        A tuple containing:
        - system_predictions: A dictionary mapping system names to their predictions.
            Only returned if `return_predictions` is True.
        - system_metrics: A nested dictionary containing computed metrics for each system.
    """
    system_predictions = {}
    system_metrics = {system: defaultdict(float) for system in systems}

    for system in tqdm(systems, desc="Evaluating model"):
        dataset = systems[system]
        preds = []
        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            past_values, future_values = zip(
                *[(data["past_values"], data["future_values"]) for data in batch]
            )
            past_batch = torch.from_numpy(np.stack(past_values)).to(model.device)
            predictions = model.predict(past_batch).transpose(1, 0).cpu().numpy()

            future_batch = np.stack(future_values)
            eval_metrics = compute_metrics(predictions, future_batch, include=metrics)

            # compute running average of metrics over batches
            for metric, value in eval_metrics.items():
                system_metrics[system][metric] += (
                    value - system_metrics[system][metric]
                ) / (i + 1)

            if return_predictions:
                preds.append(predictions)

        if return_predictions:
            system_predictions[system] = np.concatenate(preds, axis=1)

    # convert defaultdicts to regular dicts
    system_metrics = {
        system: dict(metrics) for system, metrics in system_metrics.items()
    }

    return system_predictions if return_predictions else None, system_metrics


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

    # get test data paths
    test_data_dir = os.path.expandvars(cfg.eval.data_path)
    test_data_dict = {}
    for system_dir in Path(test_data_dir).iterdir():
        if system_dir.is_dir():
            system_name = system_dir.name
            system_files = list(system_dir.glob("*"))
            test_data_dict[system_name] = [
                FileDataset(path=Path(file_path), freq="h", one_dim_target=False)
                for file_path in system_files
                if file_path.is_file()
            ]

    log_on_main(f"Running evaluation on {list(test_data_dict.keys())}", logger)

    test_datasets = {
        system_name: PatchTSTDataset(
            datasets=test_data_dict[system_name],
            probabilities=[1.0 / len(test_data_dict[system_name])]
            * len(test_data_dict[system_name]),
            context_length=cfg.patchtst.context_length,
            prediction_length=cfg.patchtst.prediction_length,
            num_test_instances=cfg.eval.num_test_instances,
            window_style=cfg.eval.window_style,
            window_stride=cfg.eval.window_stride,
            mode="test",
        )
        for system_name in test_data_dict
    }

    model = PatchTST(
        cfg.patchtst,
        mode="predict",
        pretrained_encoder_path=cfg.patchtst.pretrained_encoder_path,
    )
    model.eval()

    predictions, metrics = evaluate_forecasting_model(
        model,
        test_datasets,
        batch_size=cfg.eval.batch_size,
        metrics=[
            "mse",
            "mae",
            "smape",
            "mape",
            "r2_score",
            "spearman",
            "pearson",
        ],
        return_predictions=False,
    )

    for system in metrics:
        print(system, metrics[system])


if __name__ == "__main__":
    main()
