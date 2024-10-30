import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import hydra
import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import Forecast, SampleForecast
from tqdm.auto import tqdm

from dystformer.patchtst.model import PatchTST
from dystformer.utils import (
    load_and_split_dataset_from_arrow,
)

logger = logging.getLogger(__name__)


def generate_sample_forecasts_patchtst(
    test_data_input: Iterable,
    model: "PatchTST",
    prediction_length: int,
    batch_size: int,
    num_samples: Optional[int] = None,
    limit_prediction_length: bool = True,
    save_path: Optional[str] = None,
    **predict_kwargs,
) -> Iterable[Forecast]:
    """
    Generates forecast samples using GluonTS batcher to batch the test instances generated from FileDataset
    Returns Forecast object https://ts.gluon.ai/stable/api/gluonts/gluonts.model.forecast.html#gluonts.model.forecast.Forecast
    """
    forecast_samples = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]
        context = torch.stack(context).to(model.device).transpose(2, 1)
        print("Context shape: ", context.shape)
        forecast_samples.append(
            model.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                limit_prediction_length=limit_prediction_length,
                **predict_kwargs,
            )
            .transpose(1, 0)
            .cpu()
            .numpy()
        )
    forecast_samples = np.concatenate(forecast_samples)
    print("Forecast Samples shape: ", forecast_samples.shape)
    if save_path is not None:
        print(f"Saving forecast samples to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, forecast_samples)

    sample_forecasts = []
    for item, ts in zip(forecast_samples, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    return sample_forecasts


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    assert isinstance(torch_dtype, torch.dtype), "invalid torch dtype"

    if not os.path.isdir(cfg.eval.data_path):
        raise FileNotFoundError(f"Directory {cfg.eval.data_path} does not exist.")
    eval_dysts_names = [
        d
        for d in os.listdir(cfg.eval.data_path)
        if os.path.isdir(os.path.join(cfg.eval.data_path, d))
    ]
    print("Eval dyst dirs: ", eval_dysts_names)

    print(
        f"Loading checkpoint from {cfg.eval.checkpoint_path} onto device: {cfg.eval.device}"
    )

    model = PatchTST.from_pretrained(
        mode="predict",
        pretrain_path=cfg.eval.checkpoint_path,
        device=cfg.eval.device,
    )
    model.eval()

    result_rows = []
    for dyst_name in tqdm(eval_dysts_names):
        dyst_dir = os.path.join(cfg.eval.data_path, dyst_name)
        print(
            f"Evaluating {dyst_name} from {dyst_dir} with prediction length {cfg.eval.prediction_length} and offset {cfg.eval.offset}"
        )

        filepaths = sorted(
            list(Path(dyst_dir).glob("*.arrow")),
            key=lambda x: int(x.stem.split("_")[0]),
        )

        metrics_all_samples = defaultdict(lambda: defaultdict(list))
        for sample_idx, filepath in tqdm(enumerate(filepaths)):
            logger.info(f"Loading sample index {sample_idx}, from {filepath}")
            test_data = load_and_split_dataset_from_arrow(
                prediction_length=cfg.eval.prediction_length,
                offset=cfg.eval.offset,
                num_rolls=cfg.eval.num_rolls,
                filepath=filepath,
                one_dim_target=False,
            )

            logger.info(
                f"Generating forecasts for {dyst_name} sample {sample_idx} with ({len(test_data.input)} time series)"
            )

            forecast_save_path = None
            if cfg.eval.forecast_save_dir:
                forecast_save_path = os.path.join(
                    cfg.eval.forecast_save_dir, dyst_name, f"{filepath.stem}.npy"
                )
                os.makedirs(os.path.dirname(forecast_save_path), exist_ok=True)

            sample_forecasts = generate_sample_forecasts_patchtst(
                test_data.input,
                model,
                cfg.eval.prediction_length,
                cfg.eval.batch_size,
                num_samples=cfg.eval.num_samples,
                limit_prediction_length=cfg.eval.limit_prediction_length,
                save_path=forecast_save_path,
            )

            print(f"Forecast samples shape: {np.array(sample_forecasts).shape}")

            # logger.info("Evaluating forecasts")

            # metrics = []
            # if test_data.input:
            #     metrics = (
            #         evaluate_forecasts(
            #             sample_forecasts,
            #             test_data=test_data,
            #             metrics=[
            #                 SMAPE(),
            #                 MASE(),
            #                 RMSE(),
            #                 MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
            #             ],
            #             batch_size=5000,
            #             axis=cfg.eval.agg_axis,
            #         )
            #         .reset_index(drop=True)
            #         .to_dict(orient="records")
            #     )

            # keys = metrics[0].keys()
            # if not all(m.keys() == keys for m in metrics):
            #     raise ValueError(
            #         "Not all dictionaries (per dim) in metrics list have the same keys."
            #     )

            # for dim_idx, metrics_per_dim in enumerate(metrics):
            #     for metric_name, metric_value in metrics_per_dim.items():
            #         metrics_all_samples[dim_idx][metric_name].append(metric_value)

        # metrics_dict = {k: dict(v) for k, v in metrics_all_samples.items()}
        # metrics_all_samples = average_nested_dict(metrics_dict)

        # if cfg.eval.agg_axis is None:
        #     assert (
        #         len(metrics_all_samples) == 1
        #     ), "Expected only one dimension for axis=None aggregation"
        #     result_rows.append(
        #         {
        #             "dataset": dyst_name,
        #             "model": cfg.eval.model_id,
        #             **metrics_all_samples[0],
        #         }
        #     )
        # elif cfg.eval.agg_axis == 1:
        #     result_rows.extend(
        #         {
        #             "dataset": dyst_name,
        #             "dimension": dim_idx,
        #             "model": cfg.eval.model_id,
        #             **metrics_all_samples[dim_idx],
        #         }
        #         for dim_idx in range(len(metrics_all_samples))
        #     )
        # else:
        #     raise ValueError(f"Invalid aggregation axis: {cfg.eval.agg_axis}")

    # results_df = (
    #     pd.DataFrame(result_rows)
    #     .rename(
    #         {
    #             "sMAPE[0.5]": "sMAPE",
    #             "MASE[0.5]": "MASE",
    #             "RMSE[mean]": "RMSE",
    #             "mean_weighted_sum_quantile_loss": "WQL",
    #         },
    #         axis="columns",
    #     )
    #     .sort_values(by="dataset")
    # )

    # metrics_path = os.path.join(cfg.eval.output_dir, cfg.eval.output_fname)
    # print("Saving metrics to: ", metrics_path)
    # os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    # if os.path.isfile(metrics_path) and not cfg.eval.overwrite:
    #     existing_df = pd.read_csv(metrics_path)
    #     results_df = pd.concat([existing_df, results_df], ignore_index=True)
    # results_df.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    main()
