#!/usr/bin/env python3
"""
BasicTS 兼容的 MATS 包装器
输入: [B, L, N, C]  输出: [B, P, N, 1]
"""
import torch
import torch.nn as nn
from src.models.struct_router import StructRouter

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
        return out                        # [B, P, N]