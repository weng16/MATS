"""
专家网络模块 (Task Expert Pool)

对应图中的 Task Expert Pool (Parallel Processing):
1. PeriodicExpert (w1): 专门处理周期性模式
2. TrendExpert (w2): 专门处理趋势性模式  
3. NoiseExpert (w3): 专门处理噪声模式
4. AbruptExpert (w4): 专门处理突变点/Breakpoint模式 (图中称Abrupt Expert)
5. GeneralExpert (w5): 通用专家处理缺失值和复杂模式

每个专家都有针对其特定模式优化的架构设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Optional


class ExpertBase(nn.Module):
    """专家基类"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 共享的输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 共享的输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D_in]
        返回: [B, L, D_out]
        """
        raise NotImplementedError


class PeriodicExpert(ExpertBase):
    """
    周期性专家
    
    专门设计用于捕获周期性模式:
    - 使用傅里叶特征增强周期检测
    - 多尺度周期分解
    - 自注意力捕获长程周期依赖
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_frequencies: int = 32,
        max_period: int = 96,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)
        
        self.num_frequencies = num_frequencies
        self.max_period = max_period
        
        # 可学习的频率嵌入
        self.freq_embedding = nn.Parameter(
            torch.randn(1, num_frequencies, hidden_dim) * 0.02
        )
        
        # 傅里叶特征投影
        self.fourier_proj = nn.Linear(num_frequencies * 2, hidden_dim)
        
        # 周期感知自注意力
        self.period_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # FFN层
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
        ])
    
    def compute_fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """计算傅里叶特征"""
        B, L, D = x.shape
        
        # 生成频率基
        frequencies = torch.arange(1, self.num_frequencies + 1, device=x.device).float()
        frequencies = frequencies / self.max_period * 2 * math.pi
        
        # 时间位置
        t = torch.arange(L, device=x.device).float().unsqueeze(1)  # [L, 1]
        
        # 计算正弦和余弦特征
        angles = t * frequencies  # [L, num_freq]
        sin_features = torch.sin(angles)  # [L, num_freq]
        cos_features = torch.cos(angles)  # [L, num_freq]
        
        # 拼接
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)  # [L, 2*num_freq]
        fourier_features = fourier_features.unsqueeze(0).expand(B, -1, -1)  # [B, L, 2*num_freq]
        
        return fourier_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D_in]
        返回: [B, L, D_out]
        """
        B, L, D = x.shape
        
        # 输入投影
        h = self.input_proj(x)  # [B, L, H]
        
        # 添加傅里叶特征
        fourier_feat = self.compute_fourier_features(x)
        fourier_embed = self.fourier_proj(fourier_feat)  # [B, L, H]
        h = h + fourier_embed
        
        # 周期感知自注意力和FFN
        layer_idx = 0
        for ffn in self.ffn_layers:
            # Self-attention
            h_normed = self.layer_norms[layer_idx](h)
            attn_out, _ = self.period_attention(h_normed, h_normed, h_normed)
            h = h + self.dropout(attn_out)
            layer_idx += 1
            
            # FFN
            h_normed = self.layer_norms[layer_idx](h)
            h = h + ffn(h_normed)
            layer_idx += 1
        
        # 输出投影
        out = self.output_proj(self.norm(h))
        
        return out


class TrendExpert(ExpertBase):
    """
    趋势性专家
    
    专门设计用于捕获趋势模式:
    - 使用可分解结构提取趋势
    - 平滑滤波减少噪声影响
    - 多尺度趋势分析
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        kernel_sizes: tuple = (3, 7, 15),
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)
        
        self.kernel_sizes = kernel_sizes
        
        # 多尺度平滑滤波器
        self.smooth_convs = nn.ModuleList([
            nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=k, padding=k//2,
                groups=hidden_dim  # Depthwise conv
            ) for k in kernel_sizes
        ])
        
        # 趋势聚合
        self.trend_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 全局趋势编码
        self.global_trend_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 局部-全局融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D_in]
        返回: [B, L, D_out]
        """
        B, L, D = x.shape
        
        # 输入投影
        h = self.input_proj(x)  # [B, L, H]
        
        # 多尺度平滑
        h_conv = rearrange(h, 'b l h -> b h l')  # [B, H, L]
        smoothed = []
        for conv in self.smooth_convs:
            s = conv(h_conv)  # [B, H, L]
            smoothed.append(rearrange(s, 'b h l -> b l h'))
        
        # 聚合多尺度趋势
        multi_scale = torch.cat(smoothed, dim=-1)  # [B, L, H*3]
        local_trend = self.trend_aggregator(multi_scale)  # [B, L, H]
        
        # 全局趋势 (LSTM)
        global_trend, _ = self.global_trend_encoder(h)  # [B, L, H]
        
        # 门控融合
        concat = torch.cat([local_trend, global_trend], dim=-1)
        gate = self.fusion_gate(concat)
        h = gate * local_trend + (1 - gate) * global_trend
        
        # 输出投影
        out = self.output_proj(self.norm(h))
        
        return out


class NoiseExpert(ExpertBase):
    """
    噪声专家
    
    专门设计用于处理噪声模式:
    - 学习噪声分布特征
    - 鲁棒性特征提取
    - 方差估计
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)
        
        # 噪声鲁棒编码器 (使用较大dropout)
        self.robust_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 2),  # 更强的dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 方差估计分支
        self.variance_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保方差为正
        )
        
        # 中值滤波模拟 (可学习)
        self.median_approx = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # 去噪重建
        self.denoiser = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D_in]
        返回: [B, L, D_out]
        """
        B, L, D = x.shape
        
        # 输入投影
        h = self.input_proj(x)  # [B, L, H]
        
        # 鲁棒编码
        h_robust = self.robust_encoder(h)
        
        # 中值滤波近似
        h_conv = rearrange(h, 'b l h -> b h l')
        h_filtered = self.median_approx(h_conv)
        h_filtered = rearrange(h_filtered, 'b h l -> b l h')
        
        # 融合
        h = h_robust + h_filtered
        
        # 去噪Transformer
        h = self.denoiser(h)
        
        # 输出投影
        out = self.output_proj(self.norm(h))
        
        return out
    
    def estimate_noise_level(self, x: torch.Tensor) -> torch.Tensor:
        """估计噪声水平"""
        h = self.input_proj(x)
        h_robust = self.robust_encoder(h)
        variance = self.variance_estimator(h_robust)
        return variance


class AbruptExpert(ExpertBase):
    """
    突变点/Abrupt专家 (图中称 Abrupt Expert, w4)
    
    专门设计用于检测和处理突变点/Breakpoint:
    - 变化点检测
    - 分段建模
    - 结构断裂处理
    
    别名: ChangePointExpert
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)
        
        # 差分编码器 - 检测变化
        self.diff_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 突变点检测器
        self.change_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出变化点概率
        )
        
        # 双向LSTM用于上下文感知
        self.context_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 分段注意力
        self.segment_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
    
    def compute_difference_features(self, h: torch.Tensor) -> torch.Tensor:
        """计算差分特征用于变化点检测"""
        # 一阶差分
        diff1 = torch.zeros_like(h)
        diff1[:, 1:, :] = h[:, 1:, :] - h[:, :-1, :]
        
        # 二阶差分
        diff2 = torch.zeros_like(h)
        diff2[:, 2:, :] = diff1[:, 2:, :] - diff1[:, 1:-1, :]
        
        return diff1, diff2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D_in]
        返回: [B, L, D_out]
        """
        B, L, D = x.shape
        
        # 输入投影
        h = self.input_proj(x)  # [B, L, H]
        
        # 计算差分特征
        diff1, diff2 = self.compute_difference_features(h)
        diff_feat = self.diff_encoder(diff1 + diff2)
        
        # 检测变化点概率
        change_prob = self.change_detector(diff_feat)  # [B, L, 1]
        
        # 上下文编码
        h_context, _ = self.context_encoder(h)  # [B, L, H]
        
        # 根据变化点概率加权
        h_weighted = h_context * (1 + change_prob)
        
        # 分段注意力
        h_segment, _ = self.segment_attention(h_weighted, h_weighted, h_weighted)
        
        # 残差连接
        h = h_context + h_segment
        
        # 输出投影
        out = self.output_proj(self.norm(h))
        
        return out
    
    def detect_change_points(self, x: torch.Tensor) -> torch.Tensor:
        """返回变化点概率"""
        h = self.input_proj(x)
        diff1, diff2 = self.compute_difference_features(h)
        diff_feat = self.diff_encoder(diff1 + diff2)
        change_prob = self.change_detector(diff_feat)
        return change_prob.squeeze(-1)


class GeneralExpert(ExpertBase):
    """
    通用专家
    
    作为兜底专家处理无法归类的复杂模式:
    - 标准Transformer架构
    - 强大的表达能力
    - 处理复杂混合模式
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)
        
        # 标准Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, 512, hidden_dim) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D_in]
        返回: [B, L, D_out]
        """
        B, L, D = x.shape
        
        # 输入投影
        h = self.input_proj(x)  # [B, L, H]
        
        # 添加位置编码
        h = h + self.pos_embed[:, :L, :]
        
        # Transformer编码
        h = self.transformer(h)
        
        # 输出投影
        out = self.output_proj(self.norm(h))
        
        return out


# 别名: ChangePointExpert -> AbruptExpert (向后兼容)
ChangePointExpert = AbruptExpert


def create_expert(
    expert_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    **kwargs
) -> ExpertBase:
    """
    工厂函数: 创建指定类型的专家
    
    对应图中的 Task Expert Pool:
    - periodic (w1)
    - trend (w2)
    - noise (w3)
    - abrupt/breakpoint (w4)
    - general (w5)
    """
    experts = {
        'periodic': PeriodicExpert,
        'trend': TrendExpert,
        'noise': NoiseExpert,
        'abrupt': AbruptExpert,
        'breakpoint': AbruptExpert,  # 别名
        'changepoint': AbruptExpert,  # 别名 (向后兼容)
        'general': GeneralExpert
    }
    
    if expert_type not in experts:
        raise ValueError(f"Unknown expert type: {expert_type}")
    
    return experts[expert_type](input_dim, output_dim, hidden_dim, **kwargs)


if __name__ == "__main__":
    # 测试代码
    B, L, D_in, D_out, H = 4, 96, 7, 7, 256
    x = torch.randn(B, L, D_in)
    
    print("Testing all experts:")
    
    for expert_type in ['periodic', 'trend', 'noise', 'abrupt', 'general']:
        expert = create_expert(expert_type, D_in, D_out, H)
        out = expert(x)
        print(f"  {expert_type:12s}: input {x.shape} -> output {out.shape}")
    
    # 测试特殊功能
    print("\nSpecial functions:")
    
    noise_expert = NoiseExpert(D_in, D_out, H)
    variance = noise_expert.estimate_noise_level(x)
    print(f"  Noise variance: {variance.shape}")
    
    abrupt_expert = AbruptExpert(D_in, D_out, H)
    change_prob = abrupt_expert.detect_change_points(x)
    print(f"  Change point prob: {change_prob.shape}")
