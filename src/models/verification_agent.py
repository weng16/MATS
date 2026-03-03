"""
Verification Agent 完整实现

对应图中右侧的 Verification Agent，包含三个验证维度:
1. Consistency: 结构一致性校验
2. Validity: 预测有效性校验
3. Uncertainty: 不确定性估计

以及:
- Inverse Consistency Check (反向一致性检查)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class VerificationAgent(nn.Module):
    """
    完整的验证智能体
    
    三个验证维度:
    - Consistency: 输入输出结构一致性
    - Validity: 预测值合理性
    - Uncertainty: 模型不确定性估计
    """
    
    def __init__(
        self,
        structure_encoder,
        hidden_dim: int = 256,
        num_patterns: int = 5,
        consistency_threshold: float = 0.3,
        validity_threshold: float = 0.5,
        uncertainty_threshold: float = 0.7
    ):
        super().__init__()
        
        self.structure_encoder = structure_encoder
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        
        # 阈值
        self.consistency_threshold = consistency_threshold
        self.validity_threshold = validity_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # ==================== Consistency 模块 ====================
        self.consistency_scorer = nn.Sequential(
            nn.Linear(num_patterns * 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # ==================== Validity 模块 ====================
        # 检查预测值是否在合理范围内
        self.validity_checker = nn.Sequential(
            nn.Linear(hidden_dim + num_patterns, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 统计范围检查器
        self.range_checker = nn.Sequential(
            nn.Linear(6, 32),  # [mean, std, min, max, skew, kurtosis]
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # ==================== Uncertainty 模块 ====================
        # 基于Monte Carlo Dropout的不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(0.3),  # MC Dropout
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2)  # [aleatoric, epistemic]
        )
        
        # ==================== Inverse Consistency Check ====================
        self.inverse_checker = nn.Sequential(
            nn.Linear(num_patterns * 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_patterns),
            nn.Tanh()  # 调整量在[-1, 1]
        )
    
    # ==================== Consistency ====================
    def check_consistency(
        self,
        input_structure: torch.Tensor,
        output_structure: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        检查输入输出结构一致性
        
        参数:
            input_structure: [B, 5] 输入的模式权重
            output_structure: [B, 5] 输出的模式权重
            
        返回:
            consistency_score: [B] 一致性分数
            passed: [B] 是否通过
        """
        # 余弦相似度
        input_norm = F.normalize(input_structure, dim=-1)
        output_norm = F.normalize(output_structure, dim=-1)
        cosine_sim = (input_norm * output_norm).sum(dim=-1)  # [B]
        
        # 学习的一致性分数
        combined = torch.cat([input_structure, output_structure], dim=-1)
        learned_score = self.consistency_scorer(combined).squeeze(-1)  # [B]
        
        # 融合
        consistency_score = 0.5 * (cosine_sim + 1) / 2 + 0.5 * learned_score
        
        # 判断是否通过
        passed = consistency_score > (1 - self.consistency_threshold)
        
        return {
            'consistency_score': consistency_score,
            'cosine_similarity': cosine_sim,
            'learned_score': learned_score,
            'passed': passed
        }
    
    # ==================== Validity ====================
    def check_validity(
        self,
        prediction: torch.Tensor,
        z: torch.Tensor,
        expert_weights: torch.Tensor,
        input_stats: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        检查预测有效性
        
        验证:
        1. 预测值统计特性是否合理
        2. 是否存在异常值
        3. 是否与输入分布一致
        """
        B, L, D = prediction.shape
        
        # 计算预测的统计量
        pred_mean = prediction.mean(dim=(1, 2))  # [B]
        pred_std = prediction.std(dim=(1, 2))
        pred_min = prediction.min(dim=1)[0].min(dim=1)[0]
        pred_max = prediction.max(dim=1)[0].max(dim=1)[0]
        
        # 偏度和峰度 (简化计算)
        centered = prediction - pred_mean.view(B, 1, 1)
        skewness = (centered ** 3).mean(dim=(1, 2)) / (pred_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).mean(dim=(1, 2)) / (pred_std ** 4 + 1e-8)
        
        # 统计特征向量
        stats_vec = torch.stack([
            pred_mean, pred_std, pred_min, pred_max, 
            skewness, kurtosis
        ], dim=-1)  # [B, 6]
        
        # 范围检查分数
        range_score = self.range_checker(stats_vec).squeeze(-1)  # [B]
        
        # 基于特征的有效性检查
        combined = torch.cat([z, expert_weights], dim=-1)
        feature_score = self.validity_checker(combined).squeeze(-1)  # [B]
        
        # 异常值检测 (IQR方法)
        q1 = torch.quantile(prediction.reshape(B, -1), 0.25, dim=1)
        q3 = torch.quantile(prediction.reshape(B, -1), 0.75, dim=1)
        iqr = q3 - q1
        outlier_ratio = ((prediction.reshape(B, -1) < (q1 - 1.5 * iqr).unsqueeze(1)) | 
                        (prediction.reshape(B, -1) > (q3 + 1.5 * iqr).unsqueeze(1))).float().mean(dim=1)
        outlier_score = 1 - outlier_ratio
        
        # 综合有效性分数
        validity_score = (range_score + feature_score + outlier_score) / 3
        
        passed = validity_score > self.validity_threshold
        
        return {
            'validity_score': validity_score,
            'range_score': range_score,
            'feature_score': feature_score,
            'outlier_score': outlier_score,
            'passed': passed,
            'statistics': {
                'mean': pred_mean,
                'std': pred_std,
                'min': pred_min,
                'max': pred_max,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
        }
    
    # ==================== Uncertainty ====================
    def estimate_uncertainty(
        self,
        z: torch.Tensor,
        num_mc_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        估计预测不确定性
        
        使用Monte Carlo Dropout估计:
        - Aleatoric uncertainty: 数据固有噪声
        - Epistemic uncertainty: 模型不确定性
        """
        B = z.shape[0]
        
        # MC采样
        self.uncertainty_estimator.train()  # 启用dropout
        
        mc_outputs = []
        for _ in range(num_mc_samples):
            out = self.uncertainty_estimator(z)  # [B, 2]
            mc_outputs.append(out)
        
        mc_outputs = torch.stack(mc_outputs, dim=0)  # [num_samples, B, 2]
        
        # Aleatoric: 平均预测的aleatoric部分
        aleatoric = F.softplus(mc_outputs[:, :, 0]).mean(dim=0)  # [B]
        
        # Epistemic: MC采样的方差
        epistemic = mc_outputs[:, :, 1].var(dim=0)  # [B]
        
        # 总不确定性
        total_uncertainty = aleatoric + epistemic
        
        # 置信度分数 (不确定性越低置信度越高)
        confidence = torch.exp(-total_uncertainty)
        
        # 是否在可接受范围
        passed = confidence > (1 - self.uncertainty_threshold)
        
        self.uncertainty_estimator.eval()  # 恢复eval模式
        
        return {
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total_uncertainty': total_uncertainty,
            'confidence': confidence,
            'passed': passed
        }
    
    # ==================== Inverse Consistency Check ====================
    def inverse_consistency_check(
        self,
        input_structure: torch.Tensor,
        output_structure: torch.Tensor,
        current_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        反向一致性检查
        
        如果输出结构与输入不一致，计算需要的调整量
        """
        # 结构差异
        diff = input_structure - output_structure  # [B, 5]
        
        # 计算调整量
        combined = torch.cat([input_structure, output_structure], dim=-1)
        adjustment = self.inverse_checker(combined)  # [B, 5]
        
        # 缩放调整量 (根据差异程度)
        diff_magnitude = diff.abs().mean(dim=-1, keepdim=True)  # [B, 1]
        scaled_adjustment = adjustment * diff_magnitude
        
        # 建议的新权重
        suggested_weights = current_weights + 0.3 * scaled_adjustment
        suggested_weights = F.softmax(suggested_weights, dim=-1)
        
        return {
            'structure_diff': diff,
            'adjustment': scaled_adjustment,
            'suggested_weights': suggested_weights,
            'diff_magnitude': diff_magnitude.squeeze(-1)
        }
    
    # ==================== 综合验证 ====================
    def forward(
        self,
        input_seq: torch.Tensor,
        output_seq: torch.Tensor,
        expert_weights: torch.Tensor,
        weight_estimator,
        num_mc_samples: int = 10
    ) -> Dict:
        """
        完整的验证流程
        
        返回三个维度的验证结果和综合判断
        """
        # 提取结构特征
        with torch.no_grad():
            input_z = self.structure_encoder(input_seq)
            output_z = self.structure_encoder(output_seq)
            
            input_structure = weight_estimator(input_z)
            output_structure = weight_estimator(output_z)
        
        # 1. Consistency 检查
        consistency_result = self.check_consistency(input_structure, output_structure)
        
        # 2. Validity 检查
        validity_result = self.check_validity(
            output_seq, output_z, expert_weights
        )
        
        # 3. Uncertainty 估计
        uncertainty_result = self.estimate_uncertainty(output_z, num_mc_samples)
        
        # 4. Inverse Consistency Check
        inverse_result = self.inverse_consistency_check(
            input_structure, output_structure, expert_weights
        )
        
        # 综合判断
        all_passed = (consistency_result['passed'] & 
                     validity_result['passed'] & 
                     uncertainty_result['passed'])
        
        # 综合置信度
        overall_confidence = (
            0.4 * consistency_result['consistency_score'] +
            0.3 * validity_result['validity_score'] +
            0.3 * uncertainty_result['confidence']
        )
        
        return {
            'consistency': consistency_result,
            'validity': validity_result,
            'uncertainty': uncertainty_result,
            'inverse_check': inverse_result,
            'all_passed': all_passed,
            'overall_confidence': overall_confidence,
            'input_structure': input_structure,
            'output_structure': output_structure
        }
    
    def should_reroute(self, verification_result: Dict) -> torch.Tensor:
        """
        决定是否需要重路由
        """
        return ~verification_result['all_passed']
    
    def get_reroute_weights(
        self,
        current_weights: torch.Tensor,
        verification_result: Dict
    ) -> torch.Tensor:
        """
        获取重路由后的权重
        """
        should_reroute = self.should_reroute(verification_result)
        
        suggested = verification_result['inverse_check']['suggested_weights']
        
        # 只对需要重路由的样本应用新权重
        new_weights = torch.where(
            should_reroute.unsqueeze(-1),
            suggested,
            current_weights
        )
        
        return new_weights


class UncertaintyCalibrator(nn.Module):
    """
    不确定性校准器
    
    使用温度缩放等方法校准模型的不确定性估计
    """
    
    def __init__(self):
        super().__init__()
        
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        logits: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        校准后的置信度
        """
        # 温度缩放
        calibrated_logits = logits / self.temperature
        confidence = F.softmax(calibrated_logits, dim=-1).max(dim=-1)[0]
        
        # 结合不确定性
        calibrated_confidence = confidence * torch.exp(-uncertainty)
        
        return calibrated_confidence


if __name__ == "__main__":
    from structure_encoder import StructureEncoder
    from weight_estimator import MultiPatternWeightEstimator
    
    print("Testing VerificationAgent...")
    
    B, L, D, H = 4, 96, 7, 256
    
    # 创建模块
    encoder = StructureEncoder(input_dim=D, hidden_dim=H)
    weight_est = MultiPatternWeightEstimator(hidden_dim=H)
    agent = VerificationAgent(encoder, hidden_dim=H)
    
    # 测试数据
    input_seq = torch.randn(B, L, D)
    output_seq = torch.randn(B, L, D)
    expert_weights = F.softmax(torch.randn(B, 5), dim=-1)
    
    # 完整验证
    result = agent(input_seq, output_seq, expert_weights, weight_est)
    
    print(f"\nConsistency score: {result['consistency']['consistency_score']}")
    print(f"Consistency passed: {result['consistency']['passed']}")
    
    print(f"\nValidity score: {result['validity']['validity_score']}")
    print(f"Validity passed: {result['validity']['passed']}")
    
    print(f"\nUncertainty - Aleatoric: {result['uncertainty']['aleatoric']}")
    print(f"Uncertainty - Epistemic: {result['uncertainty']['epistemic']}")
    print(f"Confidence: {result['uncertainty']['confidence']}")
    
    print(f"\nAll passed: {result['all_passed']}")
    print(f"Overall confidence: {result['overall_confidence']}")
    
    # 测试重路由
    should_reroute = agent.should_reroute(result)
    print(f"\nShould reroute: {should_reroute}")
    
    if should_reroute.any():
        new_weights = agent.get_reroute_weights(expert_weights, result)
        print(f"Original weights: {expert_weights[0]}")
        print(f"New weights: {new_weights[0]}")
