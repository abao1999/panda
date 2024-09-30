"""
PatchTST model with the forward pass exposed
"""

from typing import Optional

import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction, PatchTSTForPretraining


class PatchTSTModel(nn.Module):
    def __init__(
        self, config: dict, mode: str = "predict", pretrain_path: Optional[str] = None
    ):
        super().__init__()

        assert mode in [
            "pretrain",
            "predict",
        ], "Mode must be either 'pretrain' or 'predict'"

        # Convert the config dict to PatchTSTConfig
        self.config = PatchTSTConfig(**config)

        if mode == "pretrain":
            self.model = PatchTSTForPretraining(self.config)
        else:  # mode == "predict"
            self.model = PatchTSTForPrediction(self.config)

        if pretrain_path is not None:
            self.load_pretrained(pretrain_path)

    def load_pretrained(self, pretrain_path: str, **kwargs):
        """
        Load a pretrained model from a path.
        """
        self.model.from_pretrained(pretrain_path, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Exposed for flexibility in channel mixing strategies.
        """
        return self.model(*args, **kwargs)
