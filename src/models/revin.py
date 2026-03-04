"""
Reversible Instance Normalization (RevIN)

Reference: Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting
against Distribution Shift", ICLR 2022.

Normalize before encoder, denormalize after prediction head.
This is a wrapper that does not alter any core model logic.
"""

import torch
import torch.nn as nn
from typing import Optional


class RevIN(nn.Module):
    """Reversible Instance Normalization for time series.

    Learns per-channel affine parameters to allow the model to
    recover the original scale after normalization.

    Args:
        num_features: Number of input channels / variables (D).
        eps: Small constant for numerical stability.
        affine: If True, learn per-channel scale (gamma) and shift (beta).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

        self._mean: Optional[torch.Tensor] = None
        self._stdev: Optional[torch.Tensor] = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input: x ∈ [B, L, D] → zero-mean unit-variance per instance per channel.

        Saves statistics for later denormalization.
        """
        self._mean = x.mean(dim=1, keepdim=True).detach()          # [B, 1, D]
        self._stdev = (x.std(dim=1, keepdim=True) + self.eps).detach()  # [B, 1, D]

        x = (x - self._mean) / self._stdev

        if self.affine:
            x = x * self.gamma + self.beta

        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization using saved statistics.

        Args:
            x: [B, L_out, D] — prediction tensor (L_out may differ from L_in).
        """
        if self._mean is None or self._stdev is None:
            return x

        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)

        x = x * self._stdev + self._mean
        return x
