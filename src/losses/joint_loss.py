"""
创新点 C7: 联合损失函数 (Joint Loss Function)

核心创新:
- 统一优化任务性能、结构一致性、稀疏性约束、因果约束
- 多目标联合优化，各组件协同学习

损失公式:
L_total = L_task + α·L_consist + β·L_sparse + γ·L_causal + δ·L_balance

各项含义:
- L_task: 下游任务损失 (预测MSE/分类CE)
- L_consist: 结构一致性损失 (保证输出结构正确)
- L_sparse: 权重稀疏性损失 (促进专家专业化)
- L_causal: DAG约束损失 (保证因果图无环)
- L_balance: 专家负载均衡 (防止专家坍塌)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TaskLoss(nn.Module):
    """
    任务损失
    
    支持多种任务类型:
    - forecasting: MSE/MAE
    - classification: CrossEntropy
    - imputation: Masked MSE
    """
    
    def __init__(
        self,
        task_type: str = 'forecasting',
        loss_type: str = 'mse',
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.task_type = task_type
        self.loss_type = loss_type
        self.reduction = reduction
        
        if task_type == 'forecasting':
            if loss_type == 'mse':
                self.criterion = nn.MSELoss(reduction=reduction)
            elif loss_type == 'mae':
                self.criterion = nn.L1Loss(reduction=reduction)
            elif loss_type == 'huber':
                self.criterion = nn.HuberLoss(reduction=reduction)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
                
        elif task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算任务损失
        
        参数:
            prediction: 预测值
            target: 目标值
            mask: 可选的掩码 (用于缺失值)
        """
        if mask is not None:
            # Masked loss
            prediction = prediction * mask
            target = target * mask
            
            if self.reduction == 'mean':
                return F.mse_loss(prediction, target, reduction='sum') / (mask.sum() + 1e-8)
            else:
                return F.mse_loss(prediction, target, reduction='none') * mask
        
        return self.criterion(prediction, target)


class StructureConsistencyLoss(nn.Module):
    """
    结构一致性损失
    
    确保预测输出的结构特征与输入一致
    
    L_consist = ||S(input) - S(output)||²
    或使用KL散度/余弦距离
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.temperature = temperature
    
    def forward(
        self,
        input_structure: torch.Tensor,
        output_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            input_structure: [B, D] 输入的结构特征
            output_structure: [B, D] 输出的结构特征
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(output_structure, input_structure)
            
        elif self.loss_type == 'cosine':
            input_norm = F.normalize(input_structure, dim=-1)
            output_norm = F.normalize(output_structure, dim=-1)
            loss = 1 - (input_norm * output_norm).sum(dim=-1).mean()
            
        elif self.loss_type == 'kl':
            # 假设是概率分布 (如权重向量)
            input_prob = F.softmax(input_structure / self.temperature, dim=-1)
            output_log_prob = F.log_softmax(output_structure / self.temperature, dim=-1)
            loss = F.kl_div(output_log_prob, input_prob, reduction='batchmean')
            
        elif self.loss_type == 'js':
            # JS散度 (对称的KL)
            input_prob = F.softmax(input_structure / self.temperature, dim=-1)
            output_prob = F.softmax(output_structure / self.temperature, dim=-1)
            m = (input_prob + output_prob) / 2
            
            kl_pm = F.kl_div(torch.log(input_prob + 1e-8), m, reduction='batchmean')
            kl_qm = F.kl_div(torch.log(output_prob + 1e-8), m, reduction='batchmean')
            loss = (kl_pm + kl_qm) / 2
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class SparsityLoss(nn.Module):
    """
    稀疏性损失
    
    鼓励权重分布有一定的集中度 (不是完全均匀)
    
    两种模式:
    - entropy: 使用熵来度量分散程度
    - l1: L1稀疏性
    """
    
    def __init__(
        self,
        mode: str = 'entropy',
        target_entropy: Optional[float] = None
    ):
        """
        参数:
            mode: 'entropy' 或 'l1'
            target_entropy: 目标熵值 (如果指定，则优化到该值)
        """
        super().__init__()
        
        self.mode = mode
        self.target_entropy = target_entropy
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        参数:
            weights: [B, K] 或 [B, L, K] 专家权重
        """
        if weights.size(-1) <= 1:
            return torch.tensor(0.0, device=weights.device)
            
        if self.mode == 'entropy':
            # 计算熵
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            
            if self.target_entropy is not None:
                # 优化到目标熵
                loss = (entropy - self.target_entropy) ** 2
            else:
                # 最小化熵 (促进集中)
                loss = entropy
                
        elif self.mode == 'l1':
            # L1稀疏性
            loss = weights.abs().sum(dim=-1).mean()
            
        elif self.mode == 'gini':
            # Gini系数 (衡量不均匀度)
            # Gini = 1 表示完全集中，Gini = 0 表示完全均匀
            sorted_weights, _ = torch.sort(weights, dim=-1)
            n = weights.shape[-1]
            indices = torch.arange(1, n + 1, device=weights.device).float()
            gini = 2 * (indices * sorted_weights).sum(dim=-1) / (n * sorted_weights.sum(dim=-1)) - (n + 1) / n
            # 我们希望有一定的Gini系数，但不要太极端
            target_gini = 0.4  # 中等集中度
            loss = (gini.mean() - target_gini) ** 2
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return loss


class DAGConstraintLoss(nn.Module):
    """
    DAG约束损失
    
    使用NOTEARS方法确保因果图是有向无环的
    
    h(W) = tr(e^{W∘W}) - N = 0
    
    当且仅当图无环时，h(W)=0
    """
    
    def __init__(self, num_agents: int = 5):
        super().__init__()
        
        self.num_agents = num_agents
    
    def forward(self, W_comm: torch.Tensor) -> torch.Tensor:
        """
        参数:
            W_comm: [N, N] 因果邻接矩阵
        """
        # 逐元素平方
        W_squared = W_comm * W_comm
        
        # 矩阵指数
        expm_W = torch.matrix_exp(W_squared)
        
        # DAG约束值
        h = torch.trace(expm_W) - self.num_agents
        
        # 使用平方使损失始终为正
        return h ** 2


class LoadBalanceLoss(nn.Module):
    """
    专家负载均衡损失
    
    防止专家坍塌，确保所有专家都被使用
    
    两种模式:
    - cv: 变异系数 (Coefficient of Variation)
    - max_ratio: 最大使用率与平均使用率之比
    """
    
    def __init__(
        self,
        num_experts: int = 5,
        mode: str = 'cv',
        auxiliary_loss: bool = True
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.mode = mode
        self.auxiliary_loss = auxiliary_loss
    
    def forward(
        self,
        weights: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        参数:
            weights: [B, K] 专家权重
            router_logits: [B, K] 路由器的原始logits (用于auxiliary loss)
        """
        if weights.size(-1) <= 1:
            return torch.tensor(0.0, device=weights.device)
            
        # 计算平均使用率
        avg_weights = weights.mean(dim=0)  # [K]
        
        if self.mode == 'cv':
            # 变异系数
            mean = avg_weights.mean()
            std = avg_weights.std()
            cv = std / (mean + 1e-8)
            loss = cv
            
        elif self.mode == 'max_ratio':
            # 最大使用率 / 平均使用率
            max_usage = avg_weights.max()
            mean_usage = avg_weights.mean()
            ratio = max_usage / (mean_usage + 1e-8)
            # 理想情况下ratio=1
            loss = (ratio - 1) ** 2
            
        elif self.mode == 'mse':
            # MSE到均匀分布
            target = torch.ones_like(avg_weights) / self.num_experts
            loss = F.mse_loss(avg_weights, target)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Auxiliary load balance loss (参考Switch Transformer)
        if self.auxiliary_loss and router_logits is not None:
            # f_i: 分配给专家i的比例
            f = weights.mean(dim=0)  # [K]
            
            # P_i: 路由器分配给专家i的概率
            P = F.softmax(router_logits, dim=-1).mean(dim=0)  # [K]
            
            # Auxiliary loss: K * Σ f_i * P_i
            aux_loss = self.num_experts * (f * P).sum()
            loss = loss + aux_loss
        
        return loss


class PrototypeOrthogonalityLoss(nn.Module):
    """Penalize deviation of prototype Gram matrix from identity.

    L_ortho = ||P_norm^T P_norm - I||²_F

    Keeps prototypes diverse throughout training, preventing mode collapse
    in the weight estimator.
    """

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prototypes: [K, H] pattern prototype matrix.
        """
        P_norm = F.normalize(prototypes, dim=-1)        # [K, H]
        gram = P_norm @ P_norm.T                        # [K, K]
        I = torch.eye(gram.size(0), device=gram.device)
        return ((gram - I) ** 2).mean()


class ExpertBalanceVarianceLoss(nn.Module):
    """Penalize variance of mean expert usage across a batch.

    L_var = Var_k( (1/B) Σ_b w_{b,k} )

    Stronger than CV for preventing collapse to a single expert.
    """

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: [B, K] expert weights per sample.
        """
        mean_usage = weights.mean(dim=0)  # [K]
        return mean_usage.var()


class JointLoss(nn.Module):
    """
    C7: Joint loss / 多目标联合损失

    L_total = L_task + α·L_consist + β·L_sparse + γ(t)·L_causal
              + δ·L_balance + ε·L_ortho

    Improvements over the original:
    - DAG penalty warmup: γ(t) linearly ramps from 0 to γ_target over
      the first ``dag_warmup_fraction`` of total steps.
    - Expert balance uses variance-of-mean-usage (stronger signal).
    - Prototype orthogonality regularization prevents prototype drift.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.01,
        gamma: float = 0.1,
        delta: float = 0.01,
        epsilon: float = 0.01,
        task_type: str = 'forecasting',
        task_loss_type: str = 'mse',
        consist_loss_type: str = 'mse',
        sparsity_mode: str = 'entropy',
        balance_mode: str = 'cv',
        num_experts: int = 5,
        dynamic_weights: bool = False,
        dag_warmup_steps: int = 0,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma_target = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.dynamic_weights = dynamic_weights

        # DAG warmup bookkeeping
        self.dag_warmup_steps = dag_warmup_steps
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

        self.task_loss = TaskLoss(task_type, task_loss_type)
        self.consist_loss = StructureConsistencyLoss(consist_loss_type)
        self.sparse_loss = SparsityLoss(sparsity_mode)
        self.dag_loss = DAGConstraintLoss(num_experts)
        self.balance_loss = LoadBalanceLoss(num_experts, balance_mode)
        self.balance_var_loss = ExpertBalanceVarianceLoss()
        self.ortho_loss = PrototypeOrthogonalityLoss()

        if dynamic_weights:
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
            self.log_beta = nn.Parameter(torch.log(torch.tensor(beta)))
            self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
            self.log_delta = nn.Parameter(torch.log(torch.tensor(delta)))

    @property
    def gamma(self) -> float:
        """Current DAG penalty weight with linear warmup."""
        if self.dag_warmup_steps <= 0:
            return self.gamma_target
        progress = min(self._step.item() / self.dag_warmup_steps, 1.0)
        return self.gamma_target * progress

    def step(self):
        """Call once per training step to advance the warmup counter."""
        self._step += 1

    def get_weights(self) -> Tuple[float, float, float, float]:
        if self.dynamic_weights:
            return (
                torch.exp(self.log_alpha).item(),
                torch.exp(self.log_beta).item(),
                torch.exp(self.log_gamma).item(),
                torch.exp(self.log_delta).item(),
            )
        return self.alpha, self.beta, self.gamma, self.delta

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        input_structure: torch.Tensor,
        output_structure: torch.Tensor,
        expert_weights: torch.Tensor,
        W_comm: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        return_components: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute the joint loss.

        Args:
            prediction: [B, L, D] model prediction.
            target: [B, L, D] ground truth.
            input_structure: [B, K] structure weights of input.
            output_structure: [B, K] structure weights of output.
            expert_weights: [B, K] per-sample expert weights.
            W_comm: [N, N] causal adjacency matrix.
            router_logits: [B, K] raw router logits (optional).
            mask: [B, L, D] loss mask for imputation (optional).
            prototypes: [K, H] pattern prototypes for orthogonality reg (optional).
            return_components: whether to return individual loss terms.
        """
        alpha, beta, gamma_static, delta = self.get_weights()
        gamma = self.gamma if not self.dynamic_weights else gamma_static

        L_task = self.task_loss(prediction, target, mask)
        L_consist = self.consist_loss(input_structure, output_structure)
        L_sparse = self.sparse_loss(expert_weights)
        L_causal = self.dag_loss(W_comm)
        L_balance = self.balance_loss(expert_weights, router_logits)
        L_balance_var = self.balance_var_loss(expert_weights)

        L_total = (
            L_task
            + alpha * L_consist
            + beta * L_sparse
            + gamma * L_causal
            + delta * (L_balance + L_balance_var)
        )

        # Prototype orthogonality regularization
        L_ortho = torch.tensor(0.0, device=prediction.device)
        if prototypes is not None:
            L_ortho = self.ortho_loss(prototypes)
            L_total = L_total + self.epsilon * L_ortho

        if return_components:
            return {
                'total': L_total,
                'task': L_task,
                'consist': L_consist,
                'sparse': L_sparse,
                'causal': L_causal,
                'balance': L_balance + L_balance_var,
                'ortho': L_ortho,
                'weights': {
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'delta': delta,
                    'epsilon': self.epsilon,
                },
            }

        return {'total': L_total}


class AdaptiveJointLoss(JointLoss):
    """
    自适应联合损失
    
    根据训练阶段自动调整各损失项权重
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 训练步数追踪
        self.register_buffer('step', torch.tensor(0))
        
        # 损失历史 (用于自适应调整)
        self.loss_history = {
            'task': [],
            'consist': [],
            'sparse': [],
            'causal': [],
            'balance': []
        }
    
    def update_step(self):
        """更新训练步数"""
        self.step += 1
    
    def get_adaptive_weights(
        self,
        current_losses: Dict[str, float]
    ) -> Tuple[float, float, float, float]:
        """
        基于当前损失值自适应调整权重
        
        策略: 损失越大的项，权重越大 (GradNorm风格)
        """
        # 更新历史
        for key in self.loss_history:
            if key in current_losses:
                self.loss_history[key].append(current_losses[key])
                # 只保留最近100个
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key] = self.loss_history[key][-100:]
        
        # 计算相对大小
        if all(len(v) > 10 for v in self.loss_history.values()):
            means = {k: sum(v[-10:]) / 10 for k, v in self.loss_history.items()}
            total = sum(means.values())
            
            # 归一化
            alpha = means['consist'] / (total + 1e-8) * 4 * self.alpha
            beta = means['sparse'] / (total + 1e-8) * 4 * self.beta
            gamma = means['causal'] / (total + 1e-8) * 4 * self.gamma
            delta = means['balance'] / (total + 1e-8) * 4 * self.delta
            
            return alpha, beta, gamma, delta
        
        return self.alpha, self.beta, self.gamma, self.delta


if __name__ == "__main__":
    # 测试代码
    B, L, D, K = 4, 96, 7, 5
    
    # 创建模拟数据
    prediction = torch.randn(B, L, D)
    target = torch.randn(B, L, D)
    input_structure = F.softmax(torch.randn(B, K), dim=-1)
    output_structure = F.softmax(torch.randn(B, K), dim=-1)
    expert_weights = F.softmax(torch.randn(B, K), dim=-1)
    W_comm = torch.randn(K, K) * 0.1
    
    print("Testing JointLoss:")
    joint_loss = JointLoss(
        alpha=0.1,
        beta=0.01,
        gamma=0.1,
        delta=0.01
    )
    
    losses = joint_loss(
        prediction, target,
        input_structure, output_structure,
        expert_weights, W_comm
    )
    
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Task loss: {losses['task'].item():.4f}")
    print(f"  Consist loss: {losses['consist'].item():.4f}")
    print(f"  Sparse loss: {losses['sparse'].item():.4f}")
    print(f"  Causal loss: {losses['causal'].item():.4f}")
    print(f"  Balance loss: {losses['balance'].item():.4f}")
    
    # 测试各组件
    print("\nTesting individual components:")
    
    task_loss = TaskLoss()
    print(f"  Task loss: {task_loss(prediction, target).item():.4f}")
    
    consist_loss = StructureConsistencyLoss(loss_type='cosine')
    print(f"  Consist loss (cosine): {consist_loss(input_structure, output_structure).item():.4f}")
    
    sparse_loss = SparsityLoss(mode='entropy')
    print(f"  Sparse loss: {sparse_loss(expert_weights).item():.4f}")
    
    dag_loss = DAGConstraintLoss()
    print(f"  DAG loss: {dag_loss(W_comm).item():.4f}")
    
    balance_loss = LoadBalanceLoss()
    print(f"  Balance loss: {balance_loss(expert_weights).item():.4f}")
