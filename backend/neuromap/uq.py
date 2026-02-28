from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference


def enable_mc_dropout(model: nn.Module) -> None:
    """Set dropout layers to train mode while keeping the rest of the model in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout),
        ):
            module.train()


@torch.inference_mode()
def mc_dropout_predict(
    model: nn.Module,
    image: torch.Tensor,
    mc_samples: int,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float = 0.25,
    tumor_channel: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run Monte Carlo dropout inference and return:
    - mean probability map
    - variance map
    - predictive entropy map
    """
    if mc_samples < 2:
        raise ValueError("mc_samples must be >= 2 for uncertainty estimation.")

    enable_mc_dropout(model)
    probs = []
    for _ in range(mc_samples):
        logits = sliding_window_inference(
            image,
            roi_size=tuple(roi_size),
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
        prob = torch.softmax(logits, dim=1)[:, tumor_channel]
        probs.append(prob)

    stacked = torch.stack(probs, dim=0)
    mean_prob = stacked.mean(dim=0)
    variance = stacked.var(dim=0, unbiased=False)

    eps = 1e-6
    entropy = -(
        mean_prob * torch.log(mean_prob + eps)
        + (1.0 - mean_prob) * torch.log(1.0 - mean_prob + eps)
    )
    return mean_prob, variance, entropy
