"""
Structure Encoder: 时序结构特征提取器

核心功能:
- 从原始时序数据中提取结构特征
- 使用 TCN (Temporal Convolutional Network) + Self-Attention
- 输出用于模式权重估计的结构向量 z ∈ R^{B×H}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class CausalConv1d(nn.Module):
    """因果卷积层，确保只使用过去信息"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        # x: [B, C, L]
        out = self.conv(x)
        # 移除右侧padding，保证因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """时序卷积网络基础块"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # 残差连接（如果通道数不匹配则投影）
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        # x: [B, C, L]
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        # x: [B, L, D]
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        
        return out


class StructureEncoder(nn.Module):
    """
    结构编码器: TCN + Self-Attention
    
    输入: 时序数据 x ∈ R^{B × L × D_in}
    输出: 结构特征 z ∈ R^{B × H}
    
    结构:
    1. 输入嵌入层
    2. TCN层提取局部时序模式
    3. Self-Attention捕获全局依赖
    4. 全局池化得到结构向量
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入嵌入
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len)
        
        # TCN层 - 使用指数增长的dilation
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout)
            )
        
        # Self-Attention层
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # 全局池化后的投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 时序统计特征提取 (增强结构感知能力)
        self.stat_proj = nn.Linear(5, hidden_dim)  # 5个统计特征
    
    def extract_statistics(self, x):
        """
        提取时序统计特征
        
        x: [B, L, D]
        返回: [B, 5] 的统计特征向量
        """
        # 聚合到单变量
        x_mean = x.mean(dim=-1)  # [B, L]
        
        # 计算统计量
        mean = x_mean.mean(dim=-1, keepdim=True)  # [B, 1]
        std = x_mean.std(dim=-1, keepdim=True)    # [B, 1]
        
        # 一阶差分的统计量 (趋势性)
        diff = x_mean[:, 1:] - x_mean[:, :-1]  # [B, L-1]
        trend = diff.mean(dim=-1, keepdim=True)
        
        # 二阶差分的统计量 (变化点)
        diff2 = diff[:, 1:] - diff[:, :-1]
        change_intensity = diff2.abs().mean(dim=-1, keepdim=True)
        
        # 自相关性估计 (周期性) - 简化版
        acf_lag1 = self._autocorr(x_mean, lag=1)
        
        stats = torch.cat([mean, std, trend, change_intensity, acf_lag1], dim=-1)
        return stats
    
    def _autocorr(self, x, lag=1):
        """计算自相关系数"""
        # x: [B, L]
        if x.size(1) <= lag:
            return torch.zeros(x.size(0), 1, device=x.device)
        
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        
        n = x.size(1) - lag
        numerator = (x_centered[:, :n] * x_centered[:, lag:]).sum(dim=-1, keepdim=True)
        denominator = (x_centered ** 2).sum(dim=-1, keepdim=True) + 1e-8
        
        return numerator / denominator
    
    def forward(self, x, return_sequence=False):
        """
        前向传播
        
        参数:
            x: [B, L, D] 输入时序
            return_sequence: 是否返回完整序列特征
            
        返回:
            z: [B, H] 结构特征向量 (默认)
            或 [B, L, H] 序列特征 (如果 return_sequence=True)
        """
        B, L, D = x.shape
        
        # 1. 输入嵌入
        h = self.input_embed(x)  # [B, L, H]
        
        # 2. 位置编码
        h = self.pos_encoding(h)
        
        # 3. TCN层
        # 转换为 [B, H, L] 用于卷积
        h = rearrange(h, 'b l h -> b h l')
        for tcn in self.tcn_layers:
            h = tcn(h)
        h = rearrange(h, 'b h l -> b l h')  # [B, L, H]
        
        # 4. Self-Attention
        attn_out = self.attention(h)
        h = self.attn_norm(h + attn_out)
        
        # 5. FFN
        ffn_out = self.ffn(h)
        h = self.ffn_norm(h + ffn_out)  # [B, L, H]
        
        if return_sequence:
            return h
        
        # 6. 全局池化
        # 使用平均池化和最大池化的组合
        h_avg = h.mean(dim=1)  # [B, H]
        h_max = h.max(dim=1)[0]  # [B, H]
        h_pooled = (h_avg + h_max) / 2
        
        # 7. 融合统计特征
        stats = self.extract_statistics(x)  # [B, 5]
        stats_embed = self.stat_proj(stats)  # [B, H]
        
        # 8. 输出投影
        z = self.output_proj(h_pooled + 0.1 * stats_embed)  # [B, H]
        
        return z
    
    def get_attention_weights(self, x):
        """获取注意力权重用于可视化"""
        B, L, D = x.shape
        
        h = self.input_embed(x)
        h = self.pos_encoding(h)
        
        h = rearrange(h, 'b l h -> b h l')
        for tcn in self.tcn_layers:
            h = tcn(h)
        h = rearrange(h, 'b h l -> b l h')
        
        # 计算注意力权重
        qkv = self.attention.qkv(h).reshape(B, L, 3, self.attention.num_heads, self.attention.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.attention.scale
        attn = F.softmax(attn, dim=-1)
        
        return attn  # [B, num_heads, L, L]


if __name__ == "__main__":
    # 测试代码
    encoder = StructureEncoder(input_dim=7, hidden_dim=256)
    x = torch.randn(4, 96, 7)  # [B, L, D]
    z = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")  # Expected: [4, 256]
    
    # 测试序列输出
    z_seq = encoder(x, return_sequence=True)
    print(f"Sequence output shape: {z_seq.shape}")  # Expected: [4, 96, 256]
