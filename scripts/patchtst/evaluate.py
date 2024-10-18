import logging
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import hydra
import numpy as np
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.evaluation.metrics import (
    abs_error,
    mape,
    mse,
    quantile_loss,
    smape,
)
from tqdm.auto import tqdm

from dystformer.patchtst.dataset import PatchTSTDataset
from dystformer.patchtst.model import PatchTST
from dystformer.utils import log_on_main

logger = logging.getLogger(__name__)


def evaluate_forecasting_model(
    model: PatchTST,
    dataset: PatchTSTDataset,
    metrics: Dict[str, Callable],
    batch_interval: int = 64,
    aggregate_across_dims: bool = True,
) -> Union[
    Dict[str, Dict[str, float]], Tuple[Dict[str, float], Dict[str, Dict[str, float]]]
]:
    """
    Evaluate the model on the test dataset and save metrics
    """
    # keep track of dynamical system dimension
    dims = []
    metrics_dict_by_dim = defaultdict(lambda: defaultdict(float))

    # form batches of past and future values by dimension in a cache
    batched_past_values_by_dim = defaultdict(list)
    batched_future_values_by_dim = defaultdict(list)

    for i, system in enumerate(tqdm(dataset, desc="Evaluating model")):
        dim = system["past_values"].shape[1]
        dims.append(dim)
        batched_past_values_by_dim[dim].append(system["past_values"])
        batched_future_values_by_dim[dim].append(system["future_values"])

        if i % batch_interval == 0 and i != 0:
            for dim in batched_past_values_by_dim:
                # shape: [batch_size, context_length, num_channels]
                past_batch = torch.from_numpy(np.stack(batched_past_values_by_dim[dim]))

                # shape: [num_parallel_samples, batch_size, prediction_length, num_channels]
                prediction_batch = model.predict(past_batch).transpose(1, 0).numpy()

                # shape: [batch_size, prediction_length, num_channels]
                future_batch = np.stack(batched_future_values_by_dim[dim])

                # cumulatively update metrics via running average
                for metric_name, metric in metrics.items():
                    metrics_dict_by_dim[metric_name][dim] += (
                        metric(prediction_batch, future_batch)
                        - metrics_dict_by_dim[metric_name][dim]
                    ) / (i + 1)

            # reset batch caches
            batched_past_values_by_dim.clear()
            batched_future_values_by_dim.clear()

    print("System dimension distribution: ", np.unique(dims, return_counts=True))

    for metric_name, metric_value in metrics_dict_by_dim.items():
        print(f"{metric_name}: {metric_value}")

    if aggregate_across_dims:
        aggregated_metrics_dict = {
            metric_name: float(np.mean([v for v in metric_values.values()]))
            for metric_name, metric_values in metrics_dict_by_dim.items()
        }
        for metric_name, metric_value in aggregated_metrics_dict.items():
            print(f"{metric_name}: {metric_value}")

        return aggregated_metrics_dict, dict(metrics_dict_by_dim)

    return dict(metrics_dict_by_dim)


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
    test_data_dir = os.path.expandvars("$WORK/data/nonstandardized_test/")
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
        num_test_instances=cfg.eval.num_test_instances,
        mode="test",
    )

    model = PatchTST.from_pretrained(
        pretrain_path=cfg.eval.checkpoint_path,
        mode="predict",
    )
    model.eval()

    metrics = {
        "rmse": lambda x, y: np.sqrt(mse(x, y)),
        "smape": smape,
        "mse": mse,
        "quantile_loss": partial(quantile_loss, q=0.5),
        "mape": mape,
        "abs_error": abs_error,
    }
    metrics_dict = evaluate_forecasting_model(
        model, test_dataset, metrics, batch_interval=128
    )


if __name__ == "__main__":
    main()
