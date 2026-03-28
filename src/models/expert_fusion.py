"""
创新点 C2: 软加权专家融合 (Soft-Weighted Expert Fusion)

核心创新:
- 用连续权重替代离散Top-K选择
- 所有专家按权重贡献，保留完整知识
- 梯度流向所有专家，训练更稳定
- 避免专家坍塌问题

公式: Output = Σ(w_i × Expert_i(x))

与Top-K硬路由的对比:
- Top-K: 只有k个专家参与，其他专家知识丢弃
- 软路由: 所有专家按权重贡献，信息保留完整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange

from .experts import (
    ExpertBase,
    PeriodicExpert,
    TrendExpert,
    NoiseExpert,
    AbruptExpert,
    ChangePointExpert,  # 别名
    GeneralExpert,
    create_expert
)


class SoftWeightedExpertFusion(nn.Module):
    """
    创新点C2: 软加权专家融合
    
    核心公式: Output = Σ(w_i × Expert_i(x))
    
    特点:
    1. 所有专家按权重贡献
    2. 梯度流向所有专家
    3. 支持专家级dropout增强泛化
    4. 支持专家输出的残差连接
    """
    
    # 对应图中的 Task Expert Pool: w1-w5
    EXPERT_TYPES = ['periodic', 'trend', 'noise', 'abrupt', 'general']
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_experts: int = 5,
        expert_dropout: float = 0.0,
        use_expert_residual: bool = True,
        shared_expert: bool = False
    ):
        """
        参数:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度
            num_experts: 专家数量 (默认5)
            expert_dropout: 专家级dropout概率
            use_expert_residual: 是否使用残差连接
            shared_expert: 是否使用共享专家
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_dropout = expert_dropout
        self.use_expert_residual = use_expert_residual
        
        if num_experts == 1:
            self.experts = nn.ModuleDict({
                'general': GeneralExpert(input_dim, output_dim, hidden_dim),
            })
        else:
            # 5个专业化专家网络 (对应图中 Task Expert Pool)
            # w1: Periodic Expert, w2: Trend Expert, w3: Noise Expert
            # w4: Abrupt Expert, w5: General Expert
            self.experts = nn.ModuleDict({
                'periodic': PeriodicExpert(input_dim, output_dim, hidden_dim),
                'trend': TrendExpert(input_dim, output_dim, hidden_dim),
                'noise': NoiseExpert(input_dim, output_dim, hidden_dim),
                'abrupt': AbruptExpert(input_dim, output_dim, hidden_dim),
                'general': GeneralExpert(input_dim, output_dim, hidden_dim),
            })
        self.expert_names = list(self.experts.keys())
        
        # 可选的共享专家
        if shared_expert:
            self.shared_expert = GeneralExpert(input_dim, output_dim, hidden_dim)
            self.shared_gate = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.shared_expert = None
        
        # 专家输出变换 (可选)
        self.output_transforms = nn.ModuleDict({
            name: nn.Linear(output_dim, output_dim)
            for name in self.expert_names
        })
        
        # 残差连接投影
        if use_expert_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
    
    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        return_expert_outputs: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: [B, L, D_in] 输入时序
            weights: [B, num_experts] 模式权重
            return_expert_outputs: 是否返回各专家的输出
            
        返回:
            fused_output: [B, L, D_out] 融合后的输出
        """
        B, L, D = x.shape
        
        # 专家级dropout (训练时)
        if self.training and self.expert_dropout > 0:
            weights = self._apply_expert_dropout(weights)
        
        # 计算各专家输出
        expert_outputs = []
        expert_outputs_dict = {}
        
        for i, name in enumerate(self.expert_names):
            expert_out = self.experts[name](x)  # [B, L, D_out]
            
            # 可选的输出变换
            expert_out = self.output_transforms[name](expert_out)
            
            expert_outputs.append(expert_out)
            expert_outputs_dict[name] = expert_out
        
        # Stack: [B, num_experts, L, D_out]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 加权融合: weights扩展为[B, num_experts, 1, 1]
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # [B, 5, 1, 1]
        
        # 加权求和: [B, L, D_out]
        fused_output = (weights_expanded * expert_outputs).sum(dim=1)
        
        # 共享专家 (如果启用)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x)
            # 根据特征计算共享专家的门控
            shared_gate = self.shared_gate(x.mean(dim=1))  # [B, 1]
            shared_gate = shared_gate.unsqueeze(1)  # [B, 1, 1]
            fused_output = fused_output + shared_gate * shared_out
        
        # 残差连接
        if self.use_expert_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            fused_output = fused_output + residual
        
        if return_expert_outputs:
            return fused_output, expert_outputs_dict
        
        return fused_output
    
    def _apply_expert_dropout(self, weights: torch.Tensor) -> torch.Tensor:
        """
        应用专家级dropout
        
        随机将某些专家的权重设为0，促进其他专家学习
        """
        if not self.training:
            return weights
        
        # 生成dropout mask
        dropout_mask = torch.bernoulli(
            torch.full_like(weights, 1 - self.expert_dropout)
        )
        
        # 应用mask
        weights = weights * dropout_mask
        
        # 重新归一化
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weights
    
    def compute_load_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失
        
        鼓励各专家被均匀使用，防止专家坍塌
        """
        # 平均权重
        avg_weights = weights.mean(dim=0)  # [num_experts]
        
        # 目标是均匀分布
        target = torch.ones_like(avg_weights) / self.num_experts
        
        # MSE损失
        return F.mse_loss(avg_weights, target)
    
    def get_expert_utilization(self, weights: torch.Tensor) -> Dict[str, float]:
        """
        获取专家利用率统计
        """
        avg_weights = weights.mean(dim=0)  # [num_experts]
        
        utilization = {}
        for i, name in enumerate(self.expert_names):
            utilization[name] = avg_weights[i].item()
        
        return utilization


class HierarchicalExpertFusion(nn.Module):
    """
    扩展: 层级专家融合
    
    两层结构:
    1. 粗粒度: 确定性/随机性专家组
    2. 细粒度: 各专家组内部融合
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        
        # 确定性专家组 (周期+趋势)
        self.deterministic_experts = nn.ModuleDict({
            'periodic': PeriodicExpert(input_dim, hidden_dim, hidden_dim),
            'trend': TrendExpert(input_dim, hidden_dim, hidden_dim),
        })
        
        # 随机性专家组 (噪声+突变)
        self.stochastic_experts = nn.ModuleDict({
            'noise': NoiseExpert(input_dim, hidden_dim, hidden_dim),
            'changepoint': ChangePointExpert(input_dim, hidden_dim, hidden_dim),
        })
        
        # 通用专家
        self.general_expert = GeneralExpert(input_dim, hidden_dim, hidden_dim)
        
        # 组间融合
        self.group_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        weights: [B, 5] = [periodic, trend, noise, changepoint, general]
        """
        B, L, D = x.shape
        
        # 确定性组
        det_periodic = self.deterministic_experts['periodic'](x)
        det_trend = self.deterministic_experts['trend'](x)
        det_output = weights[:, 0:1].unsqueeze(-1) * det_periodic + \
                     weights[:, 1:2].unsqueeze(-1) * det_trend
        
        # 随机性组
        sto_noise = self.stochastic_experts['noise'](x)
        sto_change = self.stochastic_experts['changepoint'](x)
        sto_output = weights[:, 2:3].unsqueeze(-1) * sto_noise + \
                     weights[:, 3:4].unsqueeze(-1) * sto_change
        
        # 通用专家
        gen_output = weights[:, 4:5].unsqueeze(-1) * self.general_expert(x)
        
        # 组间融合
        combined = torch.cat([det_output, sto_output, gen_output], dim=-1)
        output = self.group_fusion(combined)
        
        return output


class AdaptiveExpertFusion(nn.Module):
    """
    扩展: 自适应专家融合
    
    根据输入动态调整融合策略:
    - 简单输入: 使用少量专家
    - 复杂输入: 使用更多专家
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        min_experts: int = 1,
        max_experts: int = 5
    ):
        super().__init__()
        
        self.base_fusion = SoftWeightedExpertFusion(
            input_dim, output_dim, hidden_dim
        )
        
        self.min_experts = min_experts
        self.max_experts = max_experts
        
        # 复杂度估计器
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        z: torch.Tensor  # 结构特征
    ) -> torch.Tensor:
        """
        基于输入复杂度动态调整专家数量
        """
        # 估计复杂度
        complexity = self.complexity_estimator(z)  # [B, 1]
        
        # 根据复杂度稀疏化权重
        # 复杂度低时，保留更少的专家
        k = (self.min_experts + 
             (self.max_experts - self.min_experts) * complexity).round().int()
        
        # 动态Top-K + 软融合
        sparse_weights = self._dynamic_topk(weights, k)
        
        return self.base_fusion(x, sparse_weights)
    
    def _dynamic_topk(
        self,
        weights: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        动态Top-K: 每个样本可以有不同的K
        """
        B = weights.shape[0]
        sparse_weights = torch.zeros_like(weights)
        
        for i in range(B):
            ki = k[i].item()
            topk_vals, topk_idx = weights[i].topk(int(ki))
            sparse_weights[i].scatter_(0, topk_idx, topk_vals)
        
        # 重新归一化
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return sparse_weights


if __name__ == "__main__":
    # 测试代码
    B, L, D_in, D_out, H = 4, 96, 7, 7, 256
    
    x = torch.randn(B, L, D_in)
    weights = F.softmax(torch.randn(B, 5), dim=-1)
    
    print("Testing SoftWeightedExpertFusion:")
    fusion = SoftWeightedExpertFusion(D_in, D_out, H)
    out = fusion(x, weights)
    print(f"  Input: {x.shape}, Weights: {weights.shape}")
    print(f"  Output: {out.shape}")
    
    # 测试返回专家输出
    out, expert_outs = fusion(x, weights, return_expert_outputs=True)
    print(f"\n  Expert outputs:")
    for name, eo in expert_outs.items():
        print(f"    {name}: {eo.shape}")
    
    # 测试负载均衡损失
    lb_loss = fusion.compute_load_balance_loss(weights)
    print(f"\n  Load balance loss: {lb_loss.item():.4f}")
    
    # 测试专家利用率
    utilization = fusion.get_expert_utilization(weights)
    print(f"\n  Expert utilization:")
    for name, util in utilization.items():
        print(f"    {name}: {util:.4f}")
    
    # 测试层级融合
    print("\nTesting HierarchicalExpertFusion:")
    h_fusion = HierarchicalExpertFusion(D_in, D_out, H)
    h_out = h_fusion(x, weights)
    print(f"  Output: {h_out.shape}")
