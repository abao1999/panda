"""
PatchTST model with the forward pass exposed
"""

from typing import Optional

import torch.nn as nn

from dystformer.patchtst.patchtst import PatchTSTConfig, PatchTSTForPretraining


class PatchTSTModel(nn.Module):
    def __init__(
        self, config: dict, mode: str = "predict", pretrain_path: Optional[str] = None
    ):
        super().__init__()

        self.mode = mode
        assert mode in [
            "pretrain",
            "predict",
        ], "Mode must be either 'pretrain' or 'predict'"

        # Convert the config dict to PatchTSTConfig
        self.config = PatchTSTConfig(**config)

        if mode == "pretrain":
            self.model = PatchTSTForPretraining(self.config)

        if pretrain_path is not None:
            self.load_pretrained(pretrain_path)

    @property
    def device(self):
        return self.model.device

    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)

    def load_pretrained(self, pretrain_path: str, **kwargs):
        """
        Load a pretrained model from a path.
        """
        self.model.from_pretrained(pretrain_path, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Exposed for flexibility in channel mixing strategies.
        """
        # pretraining mode does require forecasts as it does MLM
        # TODO: move this logic to the dataset to optimize forward pass cost
        if self.mode == "pretrain":
            kwargs.pop("future_values")

        return self.model(*args, **kwargs)
