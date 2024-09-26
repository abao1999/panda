import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from dystformer.chronos.pipeline import ChronosPipeline
from dystformer.utils import (
    generate_sample_forecasts,
    load_and_split_dataset_from_arrow,
)

WORK_DIR = os.getenv("WORK", "")


class ChronosForecast:
    """
    A wrapper around the Chronos forecast model class that makes it easier to use in
    forecasting tasks.

    Attributes:
        model (str): The model size to use. One of "tiny", "mini", "small", "base", "large".
        n_samples (int): The number of samples to use when making predictions.
        use_gpu (bool): Whether to use a GPU for prediction.
        max_chunk (int): The maximum number of data points to predict in one go. If the
            prediction length is greater than this value, we use the autoregressive
            mode to predict the future values.
    """

    def __init__(
        self,
        model="base",
        n_samples=20,
        max_chunk=64,
        use_gpu=True,
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.use_gpu = use_gpu
        self.max_chunk = max_chunk

        ## If a GPU is available, use it
        if self.use_gpu:
            has_gpu = torch.cuda.is_available()
            print("has gpu: ", torch.cuda.is_available(), flush=True)
            n = torch.cuda.device_count()
            print(f"{n} devices found.", flush=True)
            if has_gpu:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        self.pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{self.model}",
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        # self.name = f"chronos-{self.model}-context{self.context}"

    def predict(
        self,
        dyst_dir: str,
        prediction_length: int = 64,
        offset: int = -64,
        num_rolls: int = 1,
        batch_size: int = 32,
        limit_prediction_length: bool = False,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        forecast_save_dir: Optional[str] = None,
    ) -> np.ndarray:
        """
        Given a time series data, use the last {self.context} timepoints of data
        to predict next value in the series.

        Args:
            dyst_dir (str): The directory containing the dyst data.
                The dyst trajectories should be stored in a subdirectory for each dyst
                    data directory (with the following structure):
                    ├── dyst_name
                    │   ├── 0_T-1024.arrow
                    │   ├── 1_T-1024..arrow
                    │   ├── ...
                    │   └── N_T-1024.arrow
                    Where N is the number of samples (instance of initial condition + parameter perturbations) from the dynamical system.
                    Each Arrow file contains the trajectories for all dimensions (x, y, z, etc) for a single sample instance.
                    The shape of the data in each Arrow file is (num_dims, T). where T is the number of time points in the trajectory.
            prediction_length (int): The number of data points to predict.
            offset (int): The number of data points to use for context when making predictions.
                            if negative, then use the whole trajectory except last offset points as context.
                            i.e. for T = 1024 and offset = -64, use the first T-64 = 960 points as context.
            num_rolls (int): Number of non-overlapping windows to split the trajectory for test instances
                             (see gluonts TestTemplate.generate_instances)
            batch_size (int): batch size to batch the test instances
                            (see gluonts batcher in generate_sample_forecasts)
            limit_prediction_length (bool): Whether to limit the prediction length.
                            If True, fails loudly when prediction_length is greater than
                            the prediction length used during training (read from model checkpoint).
            temperature (Optional[float]): Temperature to use for generating sample tokens.
            top_k (Optional[int]): Top-k parameter to use for generating sample tokens.
            top_p (Optional[int]): Top-p parameter to use for generating sample tokens.
            forecast_save_dir (Optional[str]): The directory to save the forecasts to.

        Returns:
            np.ndarray: The forecasts for the time series data.
        """

        # NOTE: uncomment this to enable switching to autoregressive mode (not yet implemented)
        # # If the prediction length is greater than self.max_chunk, we need to use the autoregressive mode to predict the future values.
        # autoregressive = (prediction_length > self.max_chunk)
        # if autoregressive:
        #     print(f"Using autoregressive mode to predict {prediction_length} data points at a time.")
        #     raise NotImplementedError("Autoregressive mode is not yet implemented.")

        # assuming that the name of the folder is the dyst name
        dyst_name = os.path.basename(dyst_dir)
        print(
            f"Generating forecasts for {dyst_name} from {dyst_dir} with prediction length {prediction_length} and offset {offset}"
        )

        # get list of all dataset Arrow files associated with dyst_name
        filepaths = sorted(
            list(Path(dyst_dir).glob("*.arrow")),
            key=lambda x: int(x.stem.split("_")[0]),
        )

        all_forecasts = []
        # generate forecasts for each sample instance of dyst
        for sample_idx, filepath in tqdm(enumerate(filepaths)):
            # load dataset test split from Arrow file
            print(f"Loading sample index {sample_idx}, from {filepath}")
            test_data = load_and_split_dataset_from_arrow(
                prediction_length=prediction_length,
                offset=offset,
                num_rolls=num_rolls,
                filepath=str(filepath),
            )

            # generate forecasts for all dimensions of a single sample instance
            print(
                f"Generating forecasts for {dyst_name} sample {sample_idx} "
                f"with {len(test_data.input)} time series (one for each dimension)"
            )

            forecast_save_path = None
            if forecast_save_dir is not None:
                forecast_save_path = os.path.join(
                    forecast_save_dir, dyst_name, f"{filepath.stem}.npy"
                )
                os.makedirs(os.path.dirname(forecast_save_path), exist_ok=True)

            sample_forecasts = generate_sample_forecasts(
                test_data.input,
                pipeline=self.pipeline,
                prediction_length=prediction_length,
                batch_size=batch_size,
                num_samples=self.n_samples,
                limit_prediction_length=limit_prediction_length,
                save_path=forecast_save_path,  # if None, then don't save
                temperature=temperature,  # not needed
                top_k=top_k,  # not needed
                top_p=top_p,  # not needed
            )
            all_forecasts.append(sample_forecasts)

        all_forecasts = np.array(all_forecasts)
        return all_forecasts


if __name__ == "__main__":
    forecast_save_dir = os.path.join(WORK_DIR, "forecasts")
    dyst_dir = os.path.join(WORK_DIR, "data/train", "Lorenz")

    model = ChronosForecast(
        model="base",
        n_samples=20,
        max_chunk=64,
        use_gpu=True,
    )

    forecasts = model.predict(
        dyst_dir=dyst_dir,
        prediction_length=64,
        offset=-64,  # use the whole trajectory except last 64 points as context
        num_rolls=1,
        batch_size=32,
        limit_prediction_length=False,
        temperature=None,
        top_k=None,
        top_p=None,
        forecast_save_dir=forecast_save_dir,
    )
