#!/usr/bin/env python3
"""
BasicTS 兼容的 MATS 包装器
输入: [B, L, N, C]  输出: [B, P, N, 1]
"""
import torch
import torch.nn as nn
from src.models.struct_router import StructRouter

class MATSArch(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.core = StructRouter(**model_args)

    def forward(self, history_data: torch.Tensor,
                future_data=None, batch_seen=0, epoch=0, train=True):
        # history_data: [B, L, N, C]
        B, L, N, C = history_data.shape
        x = history_data[..., 0]          # 取第一通道 [B, L, N]
        out = self.core(x, return_details=False)['prediction']  # [B, P, N]
        return out.unsqueeze(-1)          # [B, P, N, 1]