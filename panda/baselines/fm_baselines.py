import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM


class TimesFMPipeline:
    def __init__(self, model_id: str, device: str, prediction_length: int):
        import timesfm

        self.device = device

        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=device,  # type: ignore
                per_core_batch_size=32,
                horizon_len=prediction_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id),
        )
        # for compatibility with panda
        self.model.device = self.model._device  # type: ignore

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int = -1,
        num_samples: int = -1,
        limit_prediction_length: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate forecasts for input sequences.

        Returns point forecasts and experimental quantile forecasts.
        """
        assert context.ndim == 2, "seqs must be of shape (batch_size, seq_len)"

        point_forecast, quantile_forecast = self.model.forecast(context, normalize=True)  # type: ignore
        # unsqueeze to add sample dimension
        return torch.from_numpy(point_forecast).unsqueeze(1)


class TimeMoePipeline:
    def __init__(self, model_path: str, device: str, torch_dtype: torch.dtype):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        num_samples: int = 1,
        limit_prediction_length: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        seqs = context.to(self.device)
        mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
        normed_seqs = (seqs - mean) / std
        # shape: (batch_size, context_length + prediction_length)
        output = self.model.generate(normed_seqs, max_new_tokens=prediction_length)
        # shape: (batch_size, prediction_length)
        normed_predictions = output[:, -prediction_length:]
        return (normed_predictions * std + mean).unsqueeze(1)


def _ensure_dynamix_on_path() -> Path:
    """Insert the DynaMix submodule on sys.path so we can import it inline."""
    repo_root = Path(__file__).resolve().parents[2]
    dynamix_root = repo_root / "external" / "dynamix"
    src_path = dynamix_root / "src"

    if not src_path.exists():
        raise FileNotFoundError(
            f"DynaMix sources not found at {src_path}. "
            "Did you run `git submodule update --init --recursive external/dynamix`?"
        )

    for path in (dynamix_root, src_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return src_path


@dataclass
class DynaMixPipeline:
    model_name: str
    device: str = "cpu"
    torch_dtype: torch.dtype = torch.float32
    preprocessing_method: Literal["pos_embedding", "zero_embedding", "delay_embedding", "delay_embedding_random"] = (
        "delay_embedding"
    )
    standardize: bool = True
    fit_nonstationary: bool = False
    mode: str = "predict"

    def __post_init__(self) -> None:
        _ensure_dynamix_on_path()
        # lazy import
        from src.model.forecaster import DynaMixForecaster  # type: ignore  # noqa: E402
        from src.utilities.utilities import load_hf_model  # type: ignore  # noqa: E402

        self.device = torch.device(self.device)  # type: ignore[assignment]
        self.model = load_hf_model(self.model_name)
        self.model = self.model.to(self.device, dtype=self.torch_dtype)
        self.model.eval()
        self.forecaster = DynaMixForecaster(self.model)

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        preprocessing_method: str | None = None,
        standardize: bool | None = None,
        fit_nonstationary: bool | None = None,
        initial_x: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.ndim != 3:
            raise ValueError("Expected context tensor with shape (batch_size, context_length, dim)")

        method = preprocessing_method or self.preprocessing_method
        do_standardize = self.standardize if standardize is None else standardize
        do_fit_nonstationary = self.fit_nonstationary if fit_nonstationary is None else fit_nonstationary

        seq = context.to(self.device, dtype=self.torch_dtype).permute(1, 0, 2)  # (context_len, batch, dim)
        preds = self.forecaster.forecast(
            context=seq,
            horizon=prediction_length,
            preprocessing_method=method,
            standardize=do_standardize,
            fit_nonstationary=do_fit_nonstationary,
            initial_x=initial_x,
        )

        return preds.permute(1, 0, 2).unsqueeze(1)
