#!/usr/bin/env python3
"""
BasicTS 兼容的 MATS 包装器
输入: [B, L, N, C]  输出: [B, P, N, 1]
"""
import torch
import torch.nn as nn
import numpy as np
from src.models.struct_router import StructRouter
from basicts.scaler.base_scaler import BasicTSScaler

class GlobalZScoreScaler(BasicTSScaler):
    """
    A custom scaler that correctly handles global mean/std scaling.
    Fixes the bug in BasicTS where `torch.Tensor(np.mean(data))` fails when mean is a scalar.
    """
    def __init__(self, norm_each_channel: bool, rescale: bool, stats: dict = None):
        # We enforce norm_each_channel=False here
        super().__init__(False, rescale, stats or {})

    def fit(self, data) -> None:
        if self.stats:
            return
        
        if isinstance(data, np.ndarray):
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                std = 1.0
            # Fix: torch.tensor instead of torch.Tensor for scalars
            self.stats['mean'] = torch.tensor(mean, dtype=torch.float32)
            self.stats['std'] = torch.tensor(std, dtype=torch.float32)
        else:
            self.stats['mean'] = torch.mean(data)
            self.stats['std'] = torch.std(data)
            if self.stats['std'] == 0:
                self.stats['std'] = torch.tensor(1.0, dtype=data.dtype, device=data.device)

    def transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mean = self.stats['mean'].to(input_data.device)
        std = self.stats['std'].to(input_data.device)
        normed_data = (input_data - mean) / std
        if mask is not None:
            normed_data = torch.where(mask, normed_data, input_data)
        return normed_data

    def inverse_transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mean = self.stats['mean'].to(input_data.device)
        std = self.stats['std'].to(input_data.device)
        denormed_data = input_data * std + mean
        if mask is not None:
            denormed_data = torch.where(mask, denormed_data, input_data)
        return denormed_data


class MATSArch(nn.Module):
    def __init__(self, cfg=None, **model_args):
        if cfg is not None:
            model_args = dict(cfg) if hasattr(cfg, "__iter__") else vars(cfg)
        super().__init__()
        self.core = StructRouter(**model_args)

    def forward(self, inputs: torch.Tensor,
                future_data=None, batch_seen=0, epoch=0, train=True):
        # inputs: [B, L, N, C]
        history_data = inputs if inputs.dim() == 4 else inputs.unsqueeze(-1)
        B, L, N, C = history_data.shape
        x = history_data[..., 0]          # 取第一通道 [B, L, N]
        out = self.core(x, return_details=False)['prediction']  # [B, P, N]
        return out.unsqueeze(-1)          # [B, P, N, 1] 适配 BasicTS 的 target 形状