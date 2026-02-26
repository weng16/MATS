"""
创新点 C1: 多模式权重估计器 (Multi-Pattern Weight Estimator)

核心创新:
- 突破单一模式假设，用权重向量 w=[w1,w2,w3,w4,w5] 刻画混合模式
- 结合 MLP 直接预测和原型相似度匹配两种方式
- 公式: w = Softmax(MLP(z) + λ · Sim(z, P))

五种基础模式 (对应图中的 Learnable Structure Encoder):
1. Periodic (周期性): 季节、日周期、周周期
2. Trend (趋势性): 上升、下降、平稳
3. Noise (噪声): 高斯噪声、异方差
4. Breakpoint (突变点): 结构断裂、regime change  
5. Missing (缺失): 缺失值、不规则采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MultiPatternWeightEstimator(nn.Module):
    """
    创新点C1: 多模式混合权重估计
    
    输入: 时序片段特征 z ∈ R^{B×H}
    输出: 模式权重 w ∈ R^{B×5}, Σw_i=1
    
    设计理由:
    - MLP(z): 学习特征到权重的非线性映射
    - Sim(z, P): 引入归纳偏置，相似结构应有相似权重
    - 两者融合提高鲁棒性和泛化能力
    """
    
    # 对应图中的5种模式: Periodicity, Trend, Noise, Breakpoint, Missing
    PATTERN_NAMES = ['periodic', 'trend', 'noise', 'breakpoint', 'missing']
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_patterns: int = 5,
        lambda_sim: float = 0.5,
        temperature: float = 1.0,
        use_gumbel: bool = False
    ):
        """
        参数:
            hidden_dim: 特征维度
            num_patterns: 模式数量 (默认5种)
            lambda_sim: 平衡MLP直接预测和原型匹配的权重
            temperature: Softmax温度参数
            use_gumbel: 是否使用Gumbel-Softmax (训练时可微采样)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        
        # 可学习模式原型 - 每种模式的"标准"特征表示
        # 初始化为正交向量以促进区分性
        self.pattern_prototypes = nn.Parameter(
            torch.randn(num_patterns, hidden_dim)
        )
        nn.init.orthogonal_(self.pattern_prototypes)
        
        # MLP直接预测权重
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_patterns)
        )
        
        # 可学习的平衡系数
        self.lambda_sim = nn.Parameter(torch.tensor(lambda_sim))
        
        # 原型到权重的可学习变换 (增加表达能力)
        self.prototype_transform = nn.Linear(num_patterns, num_patterns, bias=False)
        nn.init.eye_(self.prototype_transform.weight)  # 初始化为单位矩阵
    
    def compute_prototype_similarity(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算特征与原型的相似度
        
        参数:
            z: [B, H] 结构特征
            
        返回:
            sim: [B, num_patterns] 相似度分数
        """
        # L2归一化
        proto_norm = F.normalize(self.pattern_prototypes, dim=-1)  # [5, H]
        z_norm = F.normalize(z, dim=-1)  # [B, H]
        
        # 余弦相似度
        sim = torch.matmul(z_norm, proto_norm.T)  # [B, 5]
        
        # 可学习变换
        sim = self.prototype_transform(sim)
        
        return sim
    
    def forward(
        self,
        z: torch.Tensor,
        hard: bool = False,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            z: [B, H] 结构特征向量
            hard: 是否使用硬选择 (推理时可选)
            return_components: 是否返回各分量
            
        返回:
            weights: [B, num_patterns] 模式权重，Σw_i=1
        """
        # 方式1: MLP直接预测
        mlp_logits = self.weight_mlp(z)  # [B, 5]
        
        # 方式2: 与原型的相似度
        sim_logits = self.compute_prototype_similarity(z)  # [B, 5]
        
        # 融合两种方式 (可学习权重)
        combined_logits = mlp_logits + F.softplus(self.lambda_sim) * sim_logits
        
        # 温度缩放
        scaled_logits = combined_logits / self.temperature
        
        # 计算权重
        if self.use_gumbel and self.training:
            # Gumbel-Softmax: 训练时可微采样
            weights = F.gumbel_softmax(scaled_logits, tau=self.temperature, hard=hard)
        else:
            weights = F.softmax(scaled_logits, dim=-1)  # [B, 5]
        
        if hard and not self.training:
            # 推理时的硬选择 (可选)
            indices = weights.argmax(dim=-1, keepdim=True)
            hard_weights = torch.zeros_like(weights).scatter_(-1, indices, 1.0)
            # 使用 straight-through estimator
            weights = weights + (hard_weights - weights).detach()
        
        if return_components:
            return weights, {
                'mlp_logits': mlp_logits,
                'sim_logits': sim_logits,
                'combined_logits': combined_logits,
                'lambda_sim': F.softplus(self.lambda_sim)
            }
        
        return weights
    
    def get_prototype_assignments(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取样本到原型的分配情况 (用于分析)
        
        返回各模式的贡献度
        """
        weights = self.forward(z)
        
        assignments = {}
        for i, name in enumerate(self.PATTERN_NAMES):
            assignments[name] = weights[:, i]
        
        return assignments
    
    def regularization_loss(self) -> torch.Tensor:
        """
        正则化损失: 鼓励原型多样性
        
        通过最小化原型间的相似度来促进区分性
        """
        proto_norm = F.normalize(self.pattern_prototypes, dim=-1)
        similarity_matrix = torch.matmul(proto_norm, proto_norm.T)
        
        # 去除对角线
        mask = ~torch.eye(self.num_patterns, dtype=torch.bool, device=similarity_matrix.device)
        off_diagonal = similarity_matrix[mask]
        
        # 最小化非对角元素的平方
        return (off_diagonal ** 2).mean()


class HierarchicalWeightEstimator(nn.Module):
    """
    扩展: 层级权重估计器
    
    支持粗粒度和细粒度的模式分解:
    - 粗粒度: Deterministic vs Stochastic
    - 细粒度: Periodic, Trend, Noise, ChangePoint, General
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 粗粒度分类: 确定性 vs 随机性
        self.coarse_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # [deterministic, stochastic]
        )
        
        # 确定性模式下的细分
        self.deterministic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # [periodic, trend]
        )
        
        # 随机性模式下的细分
        self.stochastic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # [noise, changepoint]
        )
        
        # General模式的权重
        self.general_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        层级权重计算
        
        返回: [B, 5] 的权重向量
        """
        # General模式的门控
        general_weight = self.general_gate(z)  # [B, 1]
        remaining_weight = 1 - general_weight
        
        # 粗粒度分类
        coarse_logits = self.coarse_classifier(z)
        coarse_weights = F.softmax(coarse_logits, dim=-1)  # [B, 2]
        
        det_weight = coarse_weights[:, 0:1]  # [B, 1]
        sto_weight = coarse_weights[:, 1:2]  # [B, 1]
        
        # 确定性模式细分
        det_logits = self.deterministic_classifier(z)
        det_sub_weights = F.softmax(det_logits, dim=-1)  # [B, 2]
        
        # 随机性模式细分
        sto_logits = self.stochastic_classifier(z)
        sto_sub_weights = F.softmax(sto_logits, dim=-1)  # [B, 2]
        
        # 组合权重
        # [periodic, trend, noise, changepoint, general]
        weights = torch.cat([
            remaining_weight * det_weight * det_sub_weights[:, 0:1],  # periodic
            remaining_weight * det_weight * det_sub_weights[:, 1:2],  # trend
            remaining_weight * sto_weight * sto_sub_weights[:, 0:1],  # noise
            remaining_weight * sto_weight * sto_sub_weights[:, 1:2],  # changepoint
            general_weight  # general
        ], dim=-1)  # [B, 5]
        
        return weights


class AdaptiveWeightEstimator(nn.Module):
    """
    扩展: 自适应权重估计器
    
    根据输入数据的置信度自适应调整权重分布的"尖锐度"
    - 高置信度: 权重集中在少数专家
    - 低置信度: 权重更均匀分布
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_patterns: int = 5,
        min_temperature: float = 0.1,
        max_temperature: float = 2.0
    ):
        super().__init__()
        
        self.base_estimator = MultiPatternWeightEstimator(
            hidden_dim=hidden_dim,
            num_patterns=num_patterns,
            temperature=1.0
        )
        
        # 温度预测网络
        self.temperature_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 输出在 [0, 1]
        )
        
        self.min_temp = min_temperature
        self.max_temp = max_temperature
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            weights: [B, num_patterns] 权重
            temperature: [B, 1] 自适应温度
        """
        # 预测自适应温度
        temp_ratio = self.temperature_predictor(z)  # [B, 1]
        temperature = self.min_temp + (self.max_temp - self.min_temp) * temp_ratio
        
        # 获取基础logits
        weights, components = self.base_estimator(z, return_components=True)
        combined_logits = components['combined_logits']
        
        # 使用自适应温度
        scaled_logits = combined_logits / temperature
        weights = F.softmax(scaled_logits, dim=-1)
        
        return weights, temperature


if __name__ == "__main__":
    # 测试代码
    batch_size = 8
    hidden_dim = 256
    
    # 测试基础权重估计器
    estimator = MultiPatternWeightEstimator(hidden_dim=hidden_dim)
    z = torch.randn(batch_size, hidden_dim)
    
    weights = estimator(z)
    print(f"Input shape: {z.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum: {weights.sum(dim=-1)}")  # 应该全为1
    print(f"Sample weights: {weights[0]}")
    
    # 测试返回组件
    weights, components = estimator(z, return_components=True)
    print(f"\nComponents:")
    print(f"  MLP logits: {components['mlp_logits'].shape}")
    print(f"  Sim logits: {components['sim_logits'].shape}")
    print(f"  Lambda sim: {components['lambda_sim'].item():.4f}")
    
    # 测试正则化损失
    reg_loss = estimator.regularization_loss()
    print(f"\nRegularization loss: {reg_loss.item():.4f}")
    
    # 测试层级估计器
    hierarchical = HierarchicalWeightEstimator(hidden_dim=hidden_dim)
    h_weights = hierarchical(z)
    print(f"\nHierarchical weights shape: {h_weights.shape}")
    print(f"Hierarchical weights sum: {h_weights.sum(dim=-1)}")
    
    # 测试自适应估计器
    adaptive = AdaptiveWeightEstimator(hidden_dim=hidden_dim)
    a_weights, temps = adaptive(z)
    print(f"\nAdaptive weights shape: {a_weights.shape}")
    print(f"Temperatures: {temps.squeeze()}")
