"""
创新点 C6: 闭环结构一致性校验 (Closed-Loop Structure Verification)

核心创新:
- 将模型输出送回结构编码器，检查输出的结构特征是否与输入一致
- 防止"幻觉预测" - 输出结构与输入不匹配
- 校验失败时触发重路由机制

设计理由:
- 输入有明显周期性 → 输出也应保持周期性
- 结构一致性是时序预测的内在约束
- 通过闭环校验提高预测可靠性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ClosedLoopVerification(nn.Module):
    """
    创新点C6: 闭环结构一致性校验
    
    防止幻觉输出，保证预测结果与输入结构特征一致
    
    流程:
    1. 提取输入结构特征 S(input)
    2. 提取输出结构特征 S(output)
    3. 计算一致性分数: 1 - ||S(input) - S(output)||²
    4. 如果分数低于阈值，触发重路由
    """
    
    def __init__(
        self,
        structure_encoder,
        threshold: float = 0.3,
        max_retry: int = 3,
        reroute_strength: float = 0.3
    ):
        """
        参数:
            structure_encoder: 结构编码器 (复用)
            threshold: 一致性阈值 (低于此值触发重路由)
            max_retry: 最大重试次数
            reroute_strength: 重路由调整强度
        """
        super().__init__()
        
        self.structure_encoder = structure_encoder
        self.threshold = threshold
        self.max_retry = max_retry
        self.reroute_strength = reroute_strength
        
        # 结构特征到权重调整的映射
        self.reroute_mapper = nn.Sequential(
            nn.Linear(5, 32),  # 5种模式
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Tanh()  # 输出在[-1, 1]，表示增减
        )
    
    def extract_structure(self, x: torch.Tensor) -> torch.Tensor:
        """
        从时序数据提取结构特征向量
        
        参数:
            x: [B, L, D] 时序数据
            
        返回:
            structure: [B, 5] 结构特征向量 (对应5种模式的强度)
        """
        with torch.no_grad():
            # 获取结构编码
            z = self.structure_encoder(x)  # [B, H]
        
        # 将高维特征投影到5维模式空间
        # 这里假设有一个weight_estimator来做这个投影
        # 实际使用时会传入weight_estimator
        return z
    
    def compute_consistency(
        self,
        input_structure: torch.Tensor,
        output_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        计算结构一致性分数
        
        参数:
            input_structure: [B, D] 输入的结构特征
            output_structure: [B, D] 输出的结构特征
            
        返回:
            score: [B] 一致性分数 (0-1)
        """
        # 使用余弦相似度
        input_norm = F.normalize(input_structure, dim=-1)
        output_norm = F.normalize(output_structure, dim=-1)
        
        cosine_sim = (input_norm * output_norm).sum(dim=-1)  # [B]
        
        # 映射到 [0, 1]
        score = (cosine_sim + 1) / 2
        
        return score
    
    def forward(
        self,
        input_seq: torch.Tensor,
        output_seq: torch.Tensor,
        expert_weights: torch.Tensor,
        weight_estimator = None
    ) -> Dict:
        """
        校验流程
        
        参数:
            input_seq: [B, L, D] 输入时序
            output_seq: [B, L_out, D] 输出时序（预测）
            expert_weights: [B, 5] 当前专家权重
            weight_estimator: 权重估计器（用于提取模式权重）
            
        返回:
            结果字典包含:
            - passed: 是否通过校验
            - consistency_score: 一致性分数
            - confidence: 预测置信度
            - input_structure: 输入结构
            - output_structure: 输出结构
        """
        # 1. 提取结构特征
        with torch.no_grad():
            input_z = self.structure_encoder(input_seq)   # [B, H]
            output_z = self.structure_encoder(output_seq) # [B, H]
        
        # 2. 如果有权重估计器，使用它来获取模式权重
        if weight_estimator is not None:
            with torch.no_grad():
                input_structure = weight_estimator(input_z)   # [B, 5]
                output_structure = weight_estimator(output_z) # [B, 5]
        else:
            # 直接使用特征（需要在外部处理）
            input_structure = input_z
            output_structure = output_z
        
        # 3. 计算一致性分数
        consistency_score = self.compute_consistency(input_structure, output_structure)
        
        # 4. 判断是否通过校验
        passed = consistency_score > (1 - self.threshold)  # [B] bool
        
        # 5. 计算置信度 (基于一致性分数和权重的集中度)
        weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(5.0))
        weight_confidence = 1 - weight_entropy / max_entropy
        
        confidence = 0.5 * consistency_score + 0.5 * weight_confidence
        
        return {
            'passed': passed,
            'consistency_score': consistency_score,
            'confidence': confidence,
            'input_structure': input_structure,
            'output_structure': output_structure,
            'input_z': input_z,
            'output_z': output_z
        }
    
    def compute_reroute_adjustment(
        self,
        input_structure: torch.Tensor,
        output_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        计算重路由的权重调整量
        
        基于输入输出结构的差异来调整权重
        """
        # 计算结构差异
        structure_diff = input_structure - output_structure  # [B, 5]
        
        # 通过学习的映射得到调整量
        adjustment = self.reroute_mapper(structure_diff)  # [B, 5]
        
        return adjustment * self.reroute_strength
    
    def reroute_on_failure(
        self,
        current_weights: torch.Tensor,
        verification_result: Dict
    ) -> torch.Tensor:
        """
        校验失败时的重路由策略
        
        策略: 增加输入结构中高权重模式对应的专家权重
        """
        input_structure = verification_result['input_structure']
        output_structure = verification_result['output_structure']
        
        # 计算调整量
        adjustment = self.compute_reroute_adjustment(input_structure, output_structure)
        
        # 应用调整 (只对未通过的样本)
        passed = verification_result['passed']
        
        # 调整权重
        adjusted_weights = current_weights.clone()
        
        # 只调整未通过校验的样本
        mask = ~passed  # [B]
        if mask.any():
            # 对未通过的样本应用调整
            adjusted_weights[mask] = adjusted_weights[mask] + adjustment[mask]
            
            # 重新归一化
            adjusted_weights = F.softmax(adjusted_weights, dim=-1)
        
        return adjusted_weights


class IterativeVerification(nn.Module):
    """
    迭代式校验: 多次校验直到通过或达到最大次数
    """
    
    def __init__(
        self,
        verification: ClosedLoopVerification,
        model_forward_fn,
        max_iterations: int = 3
    ):
        super().__init__()
        
        self.verification = verification
        self.model_forward_fn = model_forward_fn
        self.max_iterations = max_iterations
    
    def forward(
        self,
        input_seq: torch.Tensor,
        initial_weights: torch.Tensor,
        weight_estimator = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        迭代校验过程
        
        返回:
            final_output: 最终预测
            info: 包含校验历史等信息
        """
        current_weights = initial_weights
        history = []
        
        for iteration in range(self.max_iterations):
            # 使用当前权重进行预测
            output_seq = self.model_forward_fn(input_seq, current_weights)
            
            # 校验
            result = self.verification(
                input_seq, output_seq, current_weights, weight_estimator
            )
            
            history.append({
                'iteration': iteration,
                'weights': current_weights.clone(),
                'consistency_score': result['consistency_score'].mean().item(),
                'passed_ratio': result['passed'].float().mean().item()
            })
            
            # 检查是否全部通过
            if result['passed'].all():
                break
            
            # 重路由
            current_weights = self.verification.reroute_on_failure(
                current_weights, result
            )
        
        # 最终预测
        final_output = self.model_forward_fn(input_seq, current_weights)
        
        return final_output, {
            'final_weights': current_weights,
            'history': history,
            'final_result': result
        }


class StructureConsistencyLoss(nn.Module):
    """
    结构一致性损失
    
    用于训练时鼓励输出保持输入的结构特征
    """
    
    def __init__(
        self,
        structure_encoder,
        weight_estimator = None,
        loss_type: str = 'mse'
    ):
        super().__init__()
        
        self.structure_encoder = structure_encoder
        self.weight_estimator = weight_estimator
        self.loss_type = loss_type
    
    def forward(
        self,
        input_seq: torch.Tensor,
        output_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        计算结构一致性损失
        """
        # 提取结构特征
        input_z = self.structure_encoder(input_seq)
        output_z = self.structure_encoder(output_seq)
        
        # 如果有权重估计器，使用模式权重
        if self.weight_estimator is not None:
            input_struct = self.weight_estimator(input_z)
            output_struct = self.weight_estimator(output_z)
        else:
            input_struct = input_z
            output_struct = output_z
        
        # 计算损失
        if self.loss_type == 'mse':
            loss = F.mse_loss(output_struct, input_struct)
        elif self.loss_type == 'cosine':
            input_norm = F.normalize(input_struct, dim=-1)
            output_norm = F.normalize(output_struct, dim=-1)
            loss = 1 - (input_norm * output_norm).sum(dim=-1).mean()
        elif self.loss_type == 'kl':
            # KL散度 (假设是概率分布)
            loss = F.kl_div(
                F.log_softmax(output_struct, dim=-1),
                F.softmax(input_struct, dim=-1),
                reduction='batchmean'
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class ConfidenceCalibration(nn.Module):
    """
    置信度校准模块
    
    基于验证结果校准预测置信度
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # 输入: [一致性分数, 权重熵, 最大权重, 最小权重, ...]
        self.calibrator = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        consistency_score: torch.Tensor,
        expert_weights: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算校准后的置信度
        """
        B = consistency_score.shape[0]
        
        # 计算权重统计
        weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=-1)
        weight_max = expert_weights.max(dim=-1)[0]
        weight_min = expert_weights.min(dim=-1)[0]
        weight_std = expert_weights.std(dim=-1)
        
        # 构建输入特征
        features = [
            consistency_score.unsqueeze(-1),  # [B, 1]
            weight_entropy.unsqueeze(-1),
            weight_max.unsqueeze(-1),
            weight_min.unsqueeze(-1),
            weight_std.unsqueeze(-1),
            expert_weights  # [B, 5]
        ]
        
        if additional_features is not None:
            features.append(additional_features)
        
        x = torch.cat(features, dim=-1)  # [B, 10+]
        
        # 如果维度不匹配，进行调整
        if x.shape[-1] != 10:
            # 简单处理: 截断或填充
            if x.shape[-1] > 10:
                x = x[:, :10]
            else:
                x = F.pad(x, (0, 10 - x.shape[-1]))
        
        confidence = self.calibrator(x)
        
        return confidence.squeeze(-1)


if __name__ == "__main__":
    # 测试代码
    from .structure_encoder import StructureEncoder
    from .weight_estimator import MultiPatternWeightEstimator
    
    B, L, D, H = 4, 96, 7, 256
    
    # 创建模块
    encoder = StructureEncoder(input_dim=D, hidden_dim=H)
    weight_est = MultiPatternWeightEstimator(hidden_dim=H)
    verification = ClosedLoopVerification(encoder)
    
    # 测试数据
    input_seq = torch.randn(B, L, D)
    output_seq = torch.randn(B, L, D)
    expert_weights = F.softmax(torch.randn(B, 5), dim=-1)
    
    print("Testing ClosedLoopVerification:")
    result = verification(input_seq, output_seq, expert_weights, weight_est)
    print(f"  Passed: {result['passed']}")
    print(f"  Consistency score: {result['consistency_score']}")
    print(f"  Confidence: {result['confidence']}")
    
    # 测试重路由
    print("\nTesting reroute:")
    new_weights = verification.reroute_on_failure(expert_weights, result)
    print(f"  Original weights: {expert_weights[0]}")
    print(f"  Adjusted weights: {new_weights[0]}")
    
    # 测试结构一致性损失
    print("\nTesting StructureConsistencyLoss:")
    consist_loss = StructureConsistencyLoss(encoder, weight_est)
    loss = consist_loss(input_seq, output_seq)
    print(f"  Consistency loss: {loss.item():.4f}")
    
    # 测试置信度校准
    print("\nTesting ConfidenceCalibration:")
    calibrator = ConfidenceCalibration()
    calibrated_conf = calibrator(result['consistency_score'], expert_weights)
    print(f"  Calibrated confidence: {calibrated_conf}")
