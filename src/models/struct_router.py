"""
StructRouter: 结构感知多智能体协作时序分析框架

对应架构图的完整实现:

┌─────────────────────────────────────────────────────────────────────┐
│ Learnable Structure Encoder (TCN-based)                              │
│ [Periodicity + Trend + Noise + Breakpoint + Missing]                │
│                          ↓                                           │
│              Learned Latent Representation                           │
│                          ↓                                           │
│ ┌─────────────────────────┬────────────────────────────┐            │
│ │   Pattern Abstraction   │  Multi-Pattern Weight      │            │
│ │   [Segment Timeline]    │  Estimator [w1..w5]        │            │
│ └─────────────────────────┴────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Cognitive Router (Core Innovation & Orchestration)                   │
│ ┌────────────────────────────────────────────────────────────────┐  │
│ │ Structure-driven Routing + Task-driven Selection               │  │
│ │              ↓                                                 │  │
│ │ Multi-Segment Route Composition → Soft-Weighted Expert Comp    │  │
│ │              ↓                                                 │  │
│ │ Task Expert Pool: [Periodic|Trend|Noise|Abrupt|General]        │  │
│ └────────────────────────────────────────────────────────────────┘  │
│                          ↓                                           │
│         Learnable Communication Matrix (W_comm)                      │
└─────────────────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────┐  ←→  ┌────────────────────────────┐
│ Segment Fusion    │      │ Verification Agent         │
│ (Boundary         │      │ [Consistency, Validity,    │
│  Smoothing)       │      │  Uncertainty]              │
└───────────────────┘      └────────────────────────────┘
                          ↓
                       Result
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .structure_encoder import StructureEncoder
from .weight_estimator import MultiPatternWeightEstimator
from .experts import (
    PeriodicExpert, TrendExpert, NoiseExpert,
    AbruptExpert, GeneralExpert
)
from .expert_fusion import SoftWeightedExpertFusion
from .tool_adapter import ToolAdapterRFT
from .causal_communication import CausalCommunicationSCM, TemporalCausalSCM
from .verification_agent import VerificationAgent
from .segment_processor import (
    SegmentDetector,
    SegmentLevelWeightEstimator,
    PatternAbstraction,
    MultiSegmentRouteComposition,
    SegmentFusion
)
from .revin import RevIN


class CognitiveRouter(nn.Module):
    """
    Cognitive Router (Core Innovation & Orchestration)
    
    整合:
    - Structure-driven Routing (Feasibility Check)
    - Task-driven Selection (Constraint Matching)
    - Multi-Segment Route Composition
    - Soft-Weighted Expert Composition
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_patterns: int = 5,
        num_tasks: int = 4
    ):
        super().__init__()
        
        self.segment_route_composer = MultiSegmentRouteComposition(
            hidden_dim=hidden_dim,
            num_patterns=num_patterns,
            num_tasks=num_tasks
        )
    
    def forward(
        self,
        z_global: torch.Tensor,
        segment_weights: List[torch.Tensor],
        task_type: int = 0
    ) -> Dict:
        return self.segment_route_composer(z_global, segment_weights, task_type)


class StructRouter(nn.Module):
    """
    StructRouter: 完整的结构感知多智能体协作框架
    
    支持多种任务:
    - Forecast (预测)
    - Imputation (插补)
    - Classification (分类)
    - Anomaly Detection (异常检测)
    """
    
    TASK_TYPES = {
        'forecast': 0,
        'imputation': 1,
        'classification': 2,
        'anomaly_detection': 3
    }
    
    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 7,
        hidden_dim: int = 256,
        seq_len: int = 96,
        pred_len: int = 96,
        num_experts: int = 5,
        num_agents: int = 5,
        max_lag: int = 3,
        max_segments: int = 10,
        use_segment_processing: bool = True,
        use_temporal_causal: bool = False,
        use_verification: bool = True,
        use_rft: bool = True,
        task_type: str = 'forecast',
        use_revin: bool = True,
        channel_independent: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_experts = num_experts
        self.num_agents = num_agents
        self.use_segment_processing = use_segment_processing
        self.use_verification = use_verification
        self.use_rft = use_rft
        self.task_type = self.TASK_TYPES.get(task_type, 0)
        self.channel_independent = channel_independent

        # RevIN: normalize before encoder, denormalize after head
        if use_revin:
            self.revin = RevIN(num_features=input_dim, affine=True)
        else:
            self.revin = None
        
        # ==================== 1. Learnable Structure Encoder ====================
        self.structure_encoder = StructureEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            kernel_size=3,
            dropout=0.1
        )
        
        # ==================== 2. Multi-Pattern Weight Estimator ====================
        self.weight_estimator = MultiPatternWeightEstimator(
            hidden_dim=hidden_dim,
            num_patterns=num_experts,
            lambda_sim=0.5
        )
        
        # ==================== 3. Segment Processing ====================
        if use_segment_processing:
            # Segment检测器
            self.segment_detector = SegmentDetector(
                hidden_dim=hidden_dim,
                max_segments=max_segments
            )
            
            # Segment级别权重估计
            self.segment_weight_estimator = SegmentLevelWeightEstimator(
                hidden_dim=hidden_dim,
                num_patterns=num_experts
            )
            
            # Pattern Abstraction
            self.pattern_abstraction = PatternAbstraction(hidden_dim=hidden_dim)
            
            # Segment Fusion
            self.segment_fusion = SegmentFusion(hidden_dim=hidden_dim)
        
        # ==================== 4. Cognitive Router ====================
        self.cognitive_router = CognitiveRouter(
            hidden_dim=hidden_dim,
            num_patterns=num_experts,
            num_tasks=4
        )
        
        # ==================== 5. Expert Fusion ====================
        self.expert_fusion = SoftWeightedExpertFusion(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_expert_residual=True
        )
        
        # ==================== 6. Tool Adapter (RFT) ====================
        if use_rft:
            self.tool_adapter = ToolAdapterRFT(
                hidden_dim=hidden_dim,
                num_patterns=num_experts
            )
        else:
            self.tool_adapter = None
        
        # ==================== 7. Causal Communication ====================
        if use_temporal_causal:
            self.communication = TemporalCausalSCM(
                num_agents=num_agents,
                hidden_dim=hidden_dim,
                max_lag=max_lag
            )
        else:
            self.communication = CausalCommunicationSCM(
                num_agents=num_agents,
                hidden_dim=hidden_dim
            )
        
        # ==================== 8. Verification Agent ====================
        if use_verification:
            self.verification_agent = VerificationAgent(
                structure_encoder=self.structure_encoder,
                hidden_dim=hidden_dim,
                num_patterns=num_experts
            )
        else:
            self.verification_agent = None
        
        # ==================== 输出层 ====================
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        # ==================== 多任务输出头 ====================
        self.task_heads = nn.ModuleDict({
            'forecast':           nn.Linear(hidden_dim, output_dim),
            'imputation':         nn.Linear(hidden_dim, output_dim),
            'classification':     nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, output_dim)
            ),
            'anomaly_detection':  nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        })

        # Stage1 自监督重建头 — 注册为子模块，避免设备不同步
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)

        if seq_len != pred_len:
            self.seq_transform = nn.Linear(seq_len, pred_len)
        else:
            self.seq_transform = None

    # ==================== W_comm 属性 ====================
    @property
    def W_comm(self) -> torch.Tensor:
        """直接暴露因果通信矩阵，供 JointLoss 使用"""
        return self.communication.W_comm
    
    def forward(
        self,
        x: torch.Tensor,
        task_type: Optional[int] = None,
        return_details: bool = True,
        use_rft: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        完整的前向传播

        参数:
            x: [B, L, D] 输入时序
            task_type: 任务类型 (0-3)
            return_details: 是否返回详细信息
            use_rft: 是否使用RFT
        """
        B, L, D = x.shape
        task = task_type if task_type is not None else self.task_type

        # ========== RevIN: instance normalization ==========
        if self.revin is not None:
            x = self.revin.normalize(x)

        # ========== Stage 1: Structure Encoding ==========
        z_global = self.structure_encoder(x)           # [B, H]
        z_seq    = self.structure_encoder(x, return_sequence=True)  # [B, L, H]

        # 全局权重估计
        global_weights = self.weight_estimator(z_global)  # [B, 5]

        # ========== Stage 2: Segment Processing ==========
        if self.use_segment_processing:
            detect_result = self.segment_detector(x, z_seq)
            segments = self.segment_detector.get_segment_boundaries(
                detect_result['change_probs'],
                detect_result['num_segments'],
                L
            )
            seg_result      = self.segment_weight_estimator(z_seq, segments)
            segment_weights  = seg_result['segment_weights']
            segment_profiles = seg_result['segment_profiles']

            abs_result     = self.pattern_abstraction(z_global, segment_weights)
            semantic_labels = abs_result['semantic_labels']

            route_result   = self.cognitive_router(z_global, segment_weights, task)
            expert_weights  = route_result['final_weights']
            router_logits   = route_result.get('task_weights')   # [B,5] 原始logits，供L_balance使用
        else:
            segments         = [[(0, L)] for _ in range(B)]
            segment_weights  = [global_weights[b:b+1] for b in range(B)]
            segment_profiles = None
            semantic_labels  = None
            expert_weights   = global_weights
            router_logits    = None

        # ========== Stage 3: RFT (可选) ==========
        if use_rft and self.tool_adapter is not None:
            adjusted_weights, log_prob, value = self.tool_adapter(
                z_global, expert_weights, deterministic=not self.training
            )
            expert_weights = adjusted_weights
        else:
            log_prob, value = None, None

        # ========== Stage 4: Expert Fusion ==========
        fused_features, expert_outputs = self.expert_fusion(
            x, expert_weights, return_expert_outputs=True
        )

        # ========== Stage 5: Causal Communication ==========
        agent_states = torch.stack(
            [expert_outputs[name].mean(dim=1) for name in self.expert_fusion.expert_names],
            dim=1
        )  # [B, N, H]
        communicated_states = self.communication(agent_states)
        comm_context  = communicated_states.mean(dim=1)          # [B, H]
        fused_features = fused_features + 0.1 * comm_context.unsqueeze(1)

        # ========== Stage 6: 多任务预测 ==========
        task_name = {v: k for k, v in self.TASK_TYPES.items()}.get(task, 'forecast')
        head = self.task_heads[task_name] if task_name in self.task_heads else self.task_heads['forecast']

        if task_name == 'classification':
            # 分类任务：时序维度做 mean pooling 后再分类
            pooled    = fused_features.mean(dim=1)          # [B, H]
            prediction = head(pooled).unsqueeze(1).expand(-1, L, -1)
        elif task_name == 'anomaly_detection':
            # 异常检测：逐时刻打分
            prediction = head(fused_features)               # [B, L, 1]
        else:
            # forecast / imputation
            prediction = head(fused_features)               # [B, L, D_out]

        if self.seq_transform is not None and task_name in ('forecast', 'imputation'):
            prediction = prediction.transpose(1, 2)
            prediction = self.seq_transform(prediction)
            prediction = prediction.transpose(1, 2)

        # ========== Stage 7: Verification ==========
        # 注意：只在 use_verification=True 且有验证器时运行
        # 训练时不使用 no_grad，让 Inverse Consistency 梯度能反传
        if self.use_verification and self.verification_agent is not None:
            verification_result = self.verification_agent(
                input_seq=x,
                output_seq=prediction,
                expert_weights=expert_weights,
                weight_estimator=self.weight_estimator
            )
            consistency_score   = verification_result['consistency']['consistency_score']
            validity_score      = verification_result['validity']['validity_score']
            uncertainty         = verification_result['uncertainty']['total_uncertainty']
            all_passed          = verification_result['all_passed']
            overall_confidence  = verification_result['overall_confidence']
            input_structure     = verification_result['input_structure']
            output_structure    = verification_result['output_structure']

            # ---- Inverse Consistency 梯度反传 ----
            # 训练时：对未通过验证的样本，用 suggested_weights 做第二次预测
            # 第二次预测的损失会反传到 weight_estimator 和 experts
            if self.training and not all_passed.all():
                new_weights  = self.verification_agent.get_reroute_weights(
                    expert_weights, verification_result
                )
                fused2       = self.expert_fusion(x, new_weights)
                comm_ctx2    = self.communication(
                    torch.stack(
                        [fused2.mean(dim=1)] * self.num_agents, dim=1
                    )
                ).mean(dim=1)
                fused2 = fused2 + 0.1 * comm_ctx2.unsqueeze(1)
                prediction2  = head(fused2)
                if self.seq_transform is not None and task_name in ('forecast', 'imputation'):
                    prediction2 = prediction2.transpose(1, 2)
                    prediction2 = self.seq_transform(prediction2)
                    prediction2 = prediction2.transpose(1, 2)

                # 用置信度作为混合系数：置信度高的样本保持原预测
                conf = overall_confidence.detach().unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
                prediction = conf * prediction + (1 - conf) * prediction2
        else:
            consistency_score  = torch.ones(B, device=x.device)
            validity_score     = torch.ones(B, device=x.device)
            uncertainty        = torch.zeros(B, device=x.device)
            all_passed         = torch.ones(B, dtype=torch.bool, device=x.device)
            overall_confidence = torch.ones(B, device=x.device)
            input_structure    = expert_weights
            output_z           = self.structure_encoder(prediction)
            output_structure   = self.weight_estimator(output_z)

        # ========== RevIN: denormalize prediction ==========
        if self.revin is not None:
            prediction = self.revin.denormalize(prediction)

        # ========== 返回结果 ==========
        result = {
            'prediction':        prediction,
            'expert_weights':    expert_weights,
            'router_logits':     router_logits,
            # Verification
            'consistency_score':  consistency_score,
            'validity_score':     validity_score,
            'uncertainty':        uncertainty,
            'all_passed':         all_passed,
            'overall_confidence': overall_confidence,
            # Structure (供 JointLoss 使用)
            'input_structure':    input_structure,
            'output_structure':   output_structure,
            # W_comm 引用 (供训练器直接获取)
            'W_comm':             self.communication.W_comm,
        }

        if return_details:
            result.update({
                'structure_features':   z_global,
                'fused_features':       fused_features,
                'expert_outputs':       expert_outputs,
                'communicated_states':  communicated_states,
                'segments':             segments,
            })
            if self.use_segment_processing:
                result['segment_weights']  = segment_weights
                result['segment_profiles'] = segment_profiles
                result['semantic_labels']  = semantic_labels
            if log_prob is not None:
                result['rft_log_prob'] = log_prob
                result['rft_value']    = value

        return result
    
    def predict(
        self,
        x: torch.Tensor,
        task_type: Optional[int] = None,
        num_reroute: int = 0
    ) -> torch.Tensor:
        """推理接口"""
        self.eval()
        
        with torch.no_grad():
            output = self.forward(x, task_type, return_details=False)
            prediction = output['prediction']
            
            if num_reroute > 0 and self.verification_agent is not None:
                best_prediction = prediction
                best_score = output['overall_confidence'].mean()
                current_weights = output['expert_weights']
                
                for _ in range(num_reroute):
                    if not output['all_passed'].all():
                        verification_result = self.verification_agent(
                            x, prediction, current_weights, self.weight_estimator
                        )
                        new_weights = self.verification_agent.get_reroute_weights(
                            current_weights, verification_result
                        )
                        
                        fused = self.expert_fusion(x, new_weights)
                        prediction = self.prediction_head(fused)
                        if self.seq_transform is not None:
                            prediction = prediction.transpose(1, 2)
                            prediction = self.seq_transform(prediction)
                            prediction = prediction.transpose(1, 2)
                        
                        score = verification_result['overall_confidence'].mean()
                        if score > best_score:
                            best_score = score
                            best_prediction = prediction
                        
                        current_weights = new_weights
                
                return best_prediction
            
            return prediction
    
    def get_causal_graph(self) -> Dict:
        """获取因果图"""
        return self.communication.get_causal_graph()
    
    def explain_prediction(self, x: torch.Tensor) -> Dict:
        """生成预测解释"""
        self.eval()
        
        with torch.no_grad():
            output = self.forward(x, return_details=True)
        
        expert_contributions = {
            name: output['expert_weights'][:, i].mean().item()
            for i, name in enumerate(self.expert_fusion.expert_names)
        }
        
        explanation = {
            'expert_contributions': expert_contributions,
            'causal_graph': self.get_causal_graph(),
            'structure_analysis': {
                'input_structure': output['input_structure'],
                'output_structure': output['output_structure'],
                'consistency_score': output['consistency_score'].mean().item(),
                'validity_score': output['validity_score'].mean().item(),
                'uncertainty': output['uncertainty'].mean().item()
            },
            'prediction': output['prediction']
        }
        
        if self.use_segment_processing and output.get('semantic_labels'):
            explanation['semantic_labels'] = output['semantic_labels']
            explanation['segment_profiles'] = output['segment_profiles']
        
        return explanation


def create_struct_router(config: Dict) -> StructRouter:
    """Factory function for creating StructRouter from a config dict."""
    return StructRouter(
        input_dim=config.get('input_dim', 7),
        output_dim=config.get('output_dim', 7),
        hidden_dim=config.get('hidden_dim', 256),
        seq_len=config.get('seq_len', 96),
        pred_len=config.get('pred_len', 96),
        num_experts=config.get('num_experts', 5),
        num_agents=config.get('num_agents', 5),
        max_lag=config.get('max_lag', 3),
        max_segments=config.get('max_segments', 10),
        use_segment_processing=config.get('use_segment_processing', True),
        use_temporal_causal=config.get('use_temporal_causal', False),
        use_verification=config.get('use_verification', True),
        use_rft=config.get('use_rft', True),
        task_type=config.get('task_type', 'forecast'),
        use_revin=config.get('use_revin', True),
        channel_independent=config.get('channel_independent', False),
    )


if __name__ == "__main__":
    print("Testing StructRouter with full framework...")
    
    model = StructRouter(
        input_dim=7,
        output_dim=7,
        hidden_dim=256,
        seq_len=96,
        pred_len=96,
        use_segment_processing=True,
        use_verification=True
    )
    
    x = torch.randn(4, 96, 7)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Prediction shape: {output['prediction'].shape}")
    print(f"Expert weights: {output['expert_weights'][0]}")
    print(f"Consistency score: {output['consistency_score']}")
    print(f"Validity score: {output['validity_score']}")
    print(f"Uncertainty: {output['uncertainty']}")
    print(f"Overall confidence: {output['overall_confidence']}")
    print(f"All passed: {output['all_passed']}")
    
    if 'semantic_labels' in output:
        print(f"Semantic labels: {output['semantic_labels']}")
    
    explanation = model.explain_prediction(x)
    print(f"\nExpert contributions: {explanation['expert_contributions']}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
