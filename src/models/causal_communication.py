"""
创新点 C4 & C5: SCM因果通信机制

C4: SCM因果通信机制 (Structural Causal Model for Agent Communication)
- 将智能体间的通信建模为结构因果模型
- 通信矩阵即为因果邻接矩阵
- DAG约束保证有向无环

C5: 时序因果扩展 (Temporal Causal Extension)
- 扩展到时序维度
- 建模跨时间步的因果关系
- 时序因果张量

为什么用因果图而非普通注意力:
- 边表示因果关系，可解释
- 无环保证信息流向清晰
- 支持do-calculus干预分析
- 支持反事实推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class CausalCommunicationSCM(nn.Module):
    """
    创新点C4: 基于SCM的因果通信机制
    
    核心:
    1. 可学习因果邻接矩阵 W_comm
    2. DAG约束保证无环 (NOTEARS方法)
    3. 支持因果推断和可解释性
    
    公式: Agent_j.state = Σ_i W_comm[i,j] × Agent_i.output + ε_j
    
    DAG约束: h(W) = tr(e^{W∘W}) - N = 0
    """
    
    def __init__(
        self,
        num_agents: int = 5,
        hidden_dim: int = 256,
        use_dag_constraint: bool = True,
        sparsity_lambda: float = 0.01
    ):
        """
        参数:
            num_agents: 智能体/专家数量
            hidden_dim: 隐藏维度
            use_dag_constraint: 是否使用DAG约束
            sparsity_lambda: 稀疏性正则化系数
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.use_dag_constraint = use_dag_constraint
        self.sparsity_lambda = sparsity_lambda
        
        # 可学习因果邻接矩阵
        # W_comm[i,j] 表示 Agent_i → Agent_j 的因果强度
        self.W_comm = nn.Parameter(
            torch.randn(num_agents, num_agents) * 0.01
        )
        
        # 消息变换
        self.message_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 消息接收变换
        self.receive_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 输出门控
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def dag_constraint(self) -> torch.Tensor:
        """
        DAG约束: h(W) = tr(e^{W∘W}) - N = 0

        使用NOTEARS方法的连续松弛:
        - 当且仅当图无环时，h(W)=0
        - clamp W_comm 防止矩阵指数爆炸
        """
        # clamp 防止 exp 数值溢出（实践中 W 幅度不超过 2 即可保持稳定）
        W_clamped = self.W_comm.clamp(-2.0, 2.0)

        # 逐元素平方 (确保非负)
        W_squared = W_clamped * W_clamped

        # 矩阵指数
        expm_W = torch.matrix_exp(W_squared)

        # 迹减去节点数
        h = torch.trace(expm_W) - self.num_agents

        return h
    
    def get_adjacency_matrix(self, threshold: float = 0.1) -> torch.Tensor:
        """
        获取离散化的邻接矩阵
        """
        W = self.W_comm.detach()
        adjacency = (W.abs() > threshold).float()
        return adjacency
    
    def forward(
        self,
        agent_states: torch.Tensor,
        return_messages: bool = False
    ) -> torch.Tensor:
        """
        前向传播: 因果通信
        
        参数:
            agent_states: [B, N, H] - N个Agent的状态
            return_messages: 是否返回通信消息
            
        返回:
            output: [B, N, H] 通信后的Agent状态
        """
        B, N, H = agent_states.shape
        
        # 1. 消息变换
        messages = self.message_transform(agent_states)  # [B, N, H]
        
        # 2. 因果传递
        # 使用 softmax 归一化因果矩阵 (沿着"发送者"维度)
        # W_normalized[i,j] = 归一化后的 Agent_i → Agent_j 强度
        W_normalized = F.softmax(self.W_comm, dim=0)  # [N, N]
        
        # 对角线置零 (避免自环)
        mask = 1 - torch.eye(N, device=self.W_comm.device)
        W_masked = W_normalized * mask
        
        # 矩阵乘法实现信息传递
        # new_state_j = Σ_i W[i,j] × message_i
        # messages: [B, N, H] -> [B, H, N]
        messages_t = messages.transpose(1, 2)  # [B, H, N]
        received_t = torch.matmul(messages_t, W_masked)  # [B, H, N]
        received = received_t.transpose(1, 2)  # [B, N, H]
        
        # 3. 接收变换
        received = self.receive_transform(received)
        
        # 4. 门控融合
        concat = torch.cat([agent_states, received], dim=-1)  # [B, N, 2H]
        gate = self.output_gate(concat)  # [B, N, H]
        
        # 5. 门控残差连接
        output = agent_states + gate * received
        
        if return_messages:
            return output, {
                'messages': messages,
                'received': received,
                'W_normalized': W_normalized,
                'gate': gate
            }
        
        return output
    
    def get_causal_graph(self) -> Dict:
        """
        返回因果图信息，用于可视化
        """
        W = self.W_comm.detach().cpu().numpy()
        
        # 阈值化得到离散图
        threshold = 0.1
        adjacency = (np.abs(W) > threshold).astype(int)
        
        # 计算边权重
        edge_weights = np.abs(W) * adjacency
        
        # 找到所有边
        edges = []
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if adjacency[i, j] > 0:
                    edges.append({
                        'from': i,
                        'to': j,
                        'weight': float(W[i, j])
                    })
        
        return {
            'adjacency': adjacency,
            'edge_weights': edge_weights,
            'edges': edges,
            'raw_weights': W
        }
    
    def monitor_causal_graph(self) -> Dict:
        """
        训练过程中监控因果图统计信息（供 Trainer 在 epoch_end 调用）

        返回:
            sparsity: 非零边比例
            dag_violation: DAG 约束值 h(W)（越接近 0 越好）
            top_edges: 权重最大的前 3 条边 [(from, to, weight), ...]
            W_norm: W_comm 的 Frobenius 范数
        """
        with torch.no_grad():
            graph    = self.get_causal_graph()
            dag_val  = self.dag_constraint().item()
            W        = self.W_comm.detach()
            W_norm   = W.norm(p='fro').item()

            # 稀疏度：有效边占所有可能边的比例
            adj      = graph['adjacency']
            n_edges  = int(adj.sum())
            max_edges = self.num_agents * (self.num_agents - 1)
            sparsity  = n_edges / max(max_edges, 1)

            # top-3 边
            edges_sorted = sorted(graph['edges'], key=lambda e: abs(e['weight']), reverse=True)
            top_edges = edges_sorted[:3]

        return {
            'sparsity':      sparsity,
            'dag_violation': dag_val,
            'top_edges':     top_edges,
            'W_norm':        W_norm,
            'num_edges':     n_edges,
        }

    def sparsity_loss(self) -> torch.Tensor:
        """
        稀疏性损失: L1正则化
        """
        return self.sparsity_lambda * self.W_comm.abs().sum()
    
    def total_regularization(self) -> torch.Tensor:
        """
        总正则化损失 = DAG约束 + 稀疏性
        """
        loss = self.sparsity_loss()
        
        if self.use_dag_constraint:
            loss = loss + self.dag_constraint()
        
        return loss


class TemporalCausalSCM(nn.Module):
    """
    创新点C5: 时序因果图
    
    扩展SCM到时序维度，建模跨时间步的因果关系
    
    公式: X_j(t+1) = Σ_i Σ_τ W[i,j,τ] × X_i(t-τ) + ε_j
    
    其中:
    - W[i,j,τ] 表示 Agent_i 在τ步前对 Agent_j 的因果影响
    - τ=0 是瞬时因果，τ>0 是滞后因果
    """
    
    def __init__(
        self,
        num_agents: int = 5,
        hidden_dim: int = 256,
        max_lag: int = 3,
        use_dag_constraint: bool = True
    ):
        """
        参数:
            num_agents: 智能体数量
            hidden_dim: 隐藏维度
            max_lag: 最大滞后步数
            use_dag_constraint: 是否对瞬时因果施加DAG约束
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.use_dag_constraint = use_dag_constraint
        
        # 时序因果张量: [N, N, max_lag]
        # W[i,j,τ] 表示 Agent_i 在τ步前对 Agent_j 的因果影响
        self.W_temporal = nn.Parameter(
            torch.randn(num_agents, num_agents, max_lag) * 0.01
        )
        
        # 瞬时因果矩阵 (单独处理以施加DAG约束)
        self.W_instant = nn.Parameter(
            torch.randn(num_agents, num_agents) * 0.01
        )
        
        # 滞后权重的可学习衰减
        self.lag_decay = nn.Parameter(torch.tensor(0.9))
        
        # 消息编码器
        self.message_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 时序注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def temporal_dag_constraint(self) -> torch.Tensor:
        """
        时序DAG约束:
        - 同时刻 (τ=0): 施加严格的无环约束
        - 跨时刻 (τ>0): 自动满足（未来不能影响过去）
        """
        # clamp 防止矩阵指数溢出
        W_clamped = self.W_instant.clamp(-2.0, 2.0)
        W_squared = W_clamped * W_clamped
        expm_W = torch.matrix_exp(W_squared)
        h = torch.trace(expm_W) - self.num_agents
        return h
    
    def forward(
        self,
        agent_states_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        时序因果通信
        
        参数:
            agent_states_seq: [B, T, N, H] - T个时间步的Agent状态
            
        返回:
            output: [B, T, N, H] 通信后的状态序列
        """
        B, T, N, H = agent_states_seq.shape
        
        outputs = []
        
        for t in range(T):
            current_states = agent_states_seq[:, t, :, :]  # [B, N, H]
            
            # 瞬时因果传递
            W_inst_norm = F.softmax(self.W_instant, dim=0)
            mask = 1 - torch.eye(N, device=self.W_instant.device)
            W_inst_masked = W_inst_norm * mask
            
            instant_messages = torch.einsum('bnh,nm->bmh', current_states, W_inst_masked)
            
            # 滞后因果传递
            lagged_messages = torch.zeros_like(current_states)
            
            for lag in range(min(t, self.max_lag)):
                past_states = agent_states_seq[:, t - lag - 1, :, :]  # [B, N, H]
                
                # 获取该滞后的因果矩阵
                W_lag = self.W_temporal[:, :, lag]
                W_lag_norm = F.softmax(W_lag, dim=0)
                
                # 应用衰减
                decay = (self.lag_decay ** (lag + 1))
                
                # 计算滞后消息
                lag_msg = torch.einsum('bnh,nm->bmh', past_states, W_lag_norm * decay)
                lagged_messages = lagged_messages + lag_msg
            
            # 融合瞬时和滞后消息
            total_messages = instant_messages + lagged_messages
            
            # 输出
            output = current_states + self.output_proj(total_messages)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)  # [B, T, N, H]
    
    def get_temporal_causal_graph(self) -> Dict:
        """
        获取时序因果图
        """
        W_inst = self.W_instant.detach().cpu().numpy()
        W_temp = self.W_temporal.detach().cpu().numpy()
        
        return {
            'instant_adjacency': (np.abs(W_inst) > 0.1).astype(int),
            'instant_weights': W_inst,
            'temporal_weights': W_temp,
            'lag_decay': self.lag_decay.item()
        }
    
    def total_regularization(self) -> torch.Tensor:
        """
        总正则化
        """
        loss = torch.tensor(0.0, device=self.W_instant.device)
        
        # DAG约束
        if self.use_dag_constraint:
            loss = loss + self.temporal_dag_constraint()
        
        # 稀疏性
        loss = loss + 0.01 * (self.W_instant.abs().sum() + self.W_temporal.abs().sum())
        
        return loss


class MultiScaleCausalCommunication(nn.Module):
    """
    扩展: 多尺度因果通信
    
    在不同时间尺度上建模因果关系
    """
    
    def __init__(
        self,
        num_agents: int = 5,
        hidden_dim: int = 256,
        scales: List[int] = [1, 4, 16]
    ):
        super().__init__()
        
        self.scales = scales
        
        # 每个尺度一个因果通信模块
        self.scale_modules = nn.ModuleList([
            CausalCommunicationSCM(num_agents, hidden_dim)
            for _ in scales
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        agent_states: torch.Tensor,
        history: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        多尺度因果通信
        """
        scale_outputs = []
        
        for i, (scale, module) in enumerate(zip(self.scales, self.scale_modules)):
            if scale == 1:
                # 当前时刻
                out = module(agent_states)
            else:
                # 使用历史信息 (如果有)
                if history is not None and len(history) >= scale:
                    # 聚合历史状态
                    hist_states = torch.stack(history[-scale:], dim=1).mean(dim=1)
                    out = module(hist_states)
                else:
                    out = module(agent_states)
            
            scale_outputs.append(out)
        
        # 融合多尺度
        concat = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(concat)
        
        return output


class CausalExplainer:
    """
    因果可解释性分析工具
    """
    
    def __init__(self, model):
        """
        参数:
            model: 包含因果通信模块的模型
        """
        self.model = model
    
    def explain_prediction(
        self,
        input_sequence: torch.Tensor,
        expert_names: List[str] = None
    ) -> Dict:
        """
        解释为什么模型做出这个预测
        
        返回:
        1. 因果路径图: 哪些Expert影响了最终预测
        2. 贡献度: 每个Expert的影响
        3. 关键时间点: 哪些时刻的信息最重要
        """
        if expert_names is None:
            expert_names = ['periodic', 'trend', 'noise', 'changepoint', 'general']
        
        # 获取因果图
        causal_graph = self.model.communication.get_causal_graph()
        
        # 分析因果路径
        critical_paths = self._find_critical_paths(
            causal_graph['adjacency'],
            causal_graph['edge_weights']
        )
        
        # 生成解释
        explanation = self._generate_explanation(critical_paths, expert_names)
        
        return {
            'causal_graph': causal_graph,
            'critical_paths': critical_paths,
            'explanation': explanation
        }
    
    def _find_critical_paths(
        self,
        adjacency: np.ndarray,
        weights: np.ndarray
    ) -> List[Dict]:
        """
        找到关键因果路径
        """
        N = adjacency.shape[0]
        paths = []
        
        # 找到所有入度高的节点
        in_degrees = adjacency.sum(axis=0)
        
        for target in range(N):
            if in_degrees[target] > 0:
                sources = np.where(adjacency[:, target] > 0)[0]
                for source in sources:
                    paths.append({
                        'source': int(source),
                        'target': int(target),
                        'weight': float(weights[source, target])
                    })
        
        # 按权重排序
        paths.sort(key=lambda x: abs(x['weight']), reverse=True)
        
        return paths
    
    def _generate_explanation(
        self,
        critical_paths: List[Dict],
        expert_names: List[str]
    ) -> str:
        """
        生成自然语言解释
        """
        if not critical_paths:
            return "No significant causal paths detected."
        
        # 构建解释
        explanations = []
        
        for path in critical_paths[:3]:  # 只取前3条路径
            source_name = expert_names[path['source']]
            target_name = expert_names[path['target']]
            weight = path['weight']
            
            if weight > 0:
                relation = "positively influences"
            else:
                relation = "negatively influences"
            
            explanations.append(
                f"{source_name.capitalize()} expert {relation} {target_name} expert "
                f"(strength: {abs(weight):.3f})"
            )
        
        return " | ".join(explanations)
    
    def compute_shapley_values(
        self,
        model,
        input_sequence: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算各专家的Shapley Value (简化版)
        
        通过逐个关闭专家来估计贡献度
        """
        # 这是一个简化实现
        # 完整的Shapley Value计算是指数复杂度的
        
        expert_names = ['periodic', 'trend', 'noise', 'changepoint', 'general']
        num_experts = len(expert_names)
        
        # 获取完整预测
        with torch.no_grad():
            full_output = model(input_sequence)
            full_loss = F.mse_loss(full_output, target).item()
        
        contributions = {}
        
        for i, name in enumerate(expert_names):
            # 关闭第i个专家
            with torch.no_grad():
                # 创建一个mask将第i个专家的权重设为0
                z = model.structure_encoder(input_sequence)
                weights = model.weight_estimator(z)
                
                # 将第i个专家权重设为0并重新归一化
                masked_weights = weights.clone()
                masked_weights[:, i] = 0
                masked_weights = masked_weights / masked_weights.sum(dim=-1, keepdim=True)
                
                # 使用masked权重进行预测
                partial_output = model.expert_fusion(input_sequence, masked_weights)
                partial_loss = F.mse_loss(partial_output, target).item()
            
            # 贡献度 = 完整loss - 缺少该专家的loss (如果正值表示有正贡献)
            contributions[name] = partial_loss - full_loss
        
        return contributions


if __name__ == "__main__":
    # 测试代码
    B, N, H = 4, 5, 256
    
    print("Testing CausalCommunicationSCM:")
    comm = CausalCommunicationSCM(num_agents=N, hidden_dim=H)
    
    agent_states = torch.randn(B, N, H)
    output = comm(agent_states)
    print(f"  Input: {agent_states.shape}")
    print(f"  Output: {output.shape}")
    
    # DAG约束
    dag_loss = comm.dag_constraint()
    print(f"  DAG constraint loss: {dag_loss.item():.4f}")
    
    # 因果图
    causal_graph = comm.get_causal_graph()
    print(f"  Edges found: {len(causal_graph['edges'])}")
    
    print("\nTesting TemporalCausalSCM:")
    T = 10
    temporal_comm = TemporalCausalSCM(num_agents=N, hidden_dim=H, max_lag=3)
    
    agent_seq = torch.randn(B, T, N, H)
    temporal_output = temporal_comm(agent_seq)
    print(f"  Input: {agent_seq.shape}")
    print(f"  Output: {temporal_output.shape}")
    
    # 时序DAG约束
    temporal_dag_loss = temporal_comm.temporal_dag_constraint()
    print(f"  Temporal DAG constraint loss: {temporal_dag_loss.item():.4f}")
    
    print("\nTesting MultiScaleCausalCommunication:")
    multi_scale = MultiScaleCausalCommunication(num_agents=N, hidden_dim=H)
    
    multi_output = multi_scale(agent_states)
    print(f"  Output: {multi_output.shape}")
