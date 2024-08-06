import torch
from typing import List

### Utils for data pipeline
def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)