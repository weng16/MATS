"""
创新点 C3: Tool Adapter + RFT (Reinforced Fine-Tuning)

核心创新:
- 将权重估计器视为可调节的"Tool"
- 使用强化学习微调路由策略
- 策略网络在预训练权重基础上做残差调整
- PPO算法优化路由决策

为什么需要RFT:
- 预训练阶段的路由策略可能不是下游任务的最优
- RL可以针对特定任务优化路由
- 使用reward信号直接优化任务性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
from collections import deque


class ToolAdapterRFT(nn.Module):
    """
    创新点C3: 强化学习微调路由策略
    
    将权重估计视为可调节的"Tool"
    使用PPO算法优化路由决策
    
    架构:
    - 策略网络: 在原始权重基础上做调整
    - 价值网络: 估计状态价值
    - Reward: 任务性能 + 结构一致性
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_patterns: int = 5,
        alpha: float = 0.1,  # 结构一致性奖励权重
        adapter_hidden: int = 64
    ):
        """
        参数:
            hidden_dim: 结构特征维度
            num_patterns: 模式/专家数量
            alpha: 结构一致性奖励的权重
            adapter_hidden: 适配器隐藏层维度
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        self.alpha = alpha
        
        # 策略网络: 在原始权重基础上做调整
        # 输入: 基础权重 [5] + 结构特征 [H]
        self.policy_adapter = nn.Sequential(
            nn.Linear(num_patterns + hidden_dim, adapter_hidden),
            nn.LayerNorm(adapter_hidden),
            nn.ReLU(),
            nn.Linear(adapter_hidden, adapter_hidden),
            nn.ReLU(),
            nn.Linear(adapter_hidden, num_patterns)
        )
        
        # 价值网络: 估计状态价值
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, adapter_hidden),
            nn.LayerNorm(adapter_hidden),
            nn.ReLU(),
            nn.Linear(adapter_hidden, adapter_hidden),
            nn.ReLU(),
            nn.Linear(adapter_hidden, 1)
        )
        
        # 策略标准差 (可学习)
        self.log_std = nn.Parameter(torch.zeros(num_patterns))
        
        # 残差连接的可学习缩放
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        z: torch.Tensor,
        base_weights: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            z: [B, H] 结构特征
            base_weights: [B, num_patterns] 基础权重 (来自权重估计器)
            deterministic: 是否使用确定性策略
            
        返回:
            adjusted_weights: [B, num_patterns] 调整后的权重
            log_prob: [B] 动作的对数概率
            value: [B, 1] 状态价值
        """
        B = z.shape[0]
        
        # 拼接输入
        adapter_input = torch.cat([base_weights, z], dim=-1)  # [B, 5+H]
        
        # 策略网络预测调整量
        weight_delta_mean = self.policy_adapter(adapter_input)  # [B, 5]
        
        # 标准差
        std = torch.exp(self.log_std).unsqueeze(0).expand(B, -1)  # [B, 5]
        
        if deterministic:
            # 确定性策略: 直接使用均值
            weight_delta = weight_delta_mean
            log_prob = torch.zeros(B, device=z.device)
        else:
            # 随机策略: 从高斯分布采样
            dist = torch.distributions.Normal(weight_delta_mean, std)
            weight_delta = dist.sample()  # [B, 5]
            log_prob = dist.log_prob(weight_delta).sum(dim=-1)  # [B]
        
        # 残差连接 + Softmax
        # 使用可学习的残差缩放
        adjusted_logits = base_weights + self.residual_scale * weight_delta
        adjusted_weights = F.softmax(adjusted_logits, dim=-1)  # [B, 5]
        
        # 价值估计
        value = self.value_head(z)  # [B, 1]
        
        return adjusted_weights, log_prob, value
    
    def evaluate_actions(
        self,
        z: torch.Tensor,
        base_weights: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的概率和熵
        
        用于PPO更新
        """
        B = z.shape[0]
        
        adapter_input = torch.cat([base_weights, z], dim=-1)
        weight_delta_mean = self.policy_adapter(adapter_input)
        std = torch.exp(self.log_std).unsqueeze(0).expand(B, -1)
        
        dist = torch.distributions.Normal(weight_delta_mean, std)
        
        # 计算动作的对数概率
        log_prob = dist.log_prob(actions).sum(dim=-1)  # [B]
        
        # 计算熵 (用于鼓励探索)
        entropy = dist.entropy().sum(dim=-1)  # [B]
        
        # 价值估计
        value = self.value_head(z)  # [B, 1]
        
        return log_prob, entropy, value


class PPOTrainer:
    """
    PPO训练器
    
    用于RFT阶段的强化学习训练
    """
    
    def __init__(
        self,
        tool_adapter: ToolAdapterRFT,
        lr: float = 1e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.adapter = tool_adapter
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.optimizer = torch.optim.Adam(tool_adapter.parameters(), lr=lr)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计 (GAE)
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        base_weights: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        PPO更新
        
        参数:
            states: [N, H] 状态 (结构特征)
            base_weights: [N, 5] 基础权重
            actions: [N, 5] 执行的动作 (权重调整)
            old_log_probs: [N] 旧策略的对数概率
            advantages: [N] 优势估计
            returns: [N] 回报
            
        返回:
            训练统计信息
        """
        N = states.shape[0]
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epochs):
            # 随机打乱
            indices = torch.randperm(N)
            
            for start in range(0, N, self.mini_batch_size):
                end = min(start + self.mini_batch_size, N)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_base_weights = base_weights[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估当前策略
                new_log_probs, entropy, values = self.adapter.evaluate_actions(
                    batch_states, batch_base_weights, batch_actions
                )
                
                # 计算ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.adapter.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        num_updates = self.ppo_epochs * (N // self.mini_batch_size + 1)
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }


class RewardComputer:
    """
    奖励计算器
    
    Reward = 任务性能 + α × 结构一致性
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        task_reward_scale: float = 1.0
    ):
        self.alpha = alpha
        self.task_reward_scale = task_reward_scale
    
    def compute_task_reward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算任务奖励 (基于预测误差)
        
        奖励 = -MSE (越小越好)
        """
        mse = F.mse_loss(prediction, target, reduction='none').mean(dim=(-1, -2))
        # 转换为正奖励: 使用负对数变换
        reward = -torch.log(mse + 1e-8) * self.task_reward_scale
        return reward
    
    def compute_consistency_reward(
        self,
        input_structure: torch.Tensor,
        output_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        计算结构一致性奖励
        
        奖励 = 1 - MSE(input_structure, output_structure)
        """
        mse = F.mse_loss(input_structure, output_structure, reduction='none').mean(dim=-1)
        reward = 1 - mse
        return reward
    
    def compute_total_reward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        input_structure: torch.Tensor,
        output_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        计算总奖励
        
        Reward = task_reward + α × consistency_reward
        """
        task_reward = self.compute_task_reward(prediction, target)
        consistency_reward = self.compute_consistency_reward(input_structure, output_structure)
        
        total_reward = task_reward + self.alpha * consistency_reward
        
        return total_reward


class ExperienceBuffer:
    """
    经验回放缓冲区
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: torch.Tensor,
        base_weight: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        log_prob: float
    ):
        """添加一条经验"""
        self.buffer.append({
            'state': state.detach().cpu(),
            'base_weight': base_weight.detach().cpu(),
            'action': action.detach().cpu(),
            'reward': reward,
            'next_state': next_state.detach().cpu(),
            'done': done,
            'log_prob': log_prob
        })
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样一批经验"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'states': torch.stack([b['state'] for b in batch]),
            'base_weights': torch.stack([b['base_weight'] for b in batch]),
            'actions': torch.stack([b['action'] for b in batch]),
            'rewards': torch.tensor([b['reward'] for b in batch]),
            'next_states': torch.stack([b['next_state'] for b in batch]),
            'dones': torch.tensor([b['done'] for b in batch], dtype=torch.float),
            'log_probs': torch.tensor([b['log_prob'] for b in batch])
        }
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """获取所有经验"""
        return {
            'states': torch.stack([b['state'] for b in self.buffer]),
            'base_weights': torch.stack([b['base_weight'] for b in self.buffer]),
            'actions': torch.stack([b['action'] for b in self.buffer]),
            'rewards': torch.tensor([b['reward'] for b in self.buffer]),
            'next_states': torch.stack([b['next_state'] for b in self.buffer]),
            'dones': torch.tensor([b['done'] for b in self.buffer], dtype=torch.float),
            'log_probs': torch.tensor([b['log_prob'] for b in self.buffer])
        }
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # 测试代码
    B, H = 8, 256
    num_patterns = 5
    
    # 测试Tool Adapter
    adapter = ToolAdapterRFT(hidden_dim=H, num_patterns=num_patterns)
    
    z = torch.randn(B, H)
    base_weights = F.softmax(torch.randn(B, num_patterns), dim=-1)
    
    # 前向传播
    adjusted_weights, log_prob, value = adapter(z, base_weights)
    print(f"Input z: {z.shape}, base_weights: {base_weights.shape}")
    print(f"Adjusted weights: {adjusted_weights.shape}")
    print(f"Log prob: {log_prob.shape}")
    print(f"Value: {value.shape}")
    print(f"Weights sum: {adjusted_weights.sum(dim=-1)}")
    
    # 确定性模式
    det_weights, _, _ = adapter(z, base_weights, deterministic=True)
    print(f"\nDeterministic weights: {det_weights[0]}")
    
    # 测试PPO训练器
    print("\nTesting PPO Trainer:")
    trainer = PPOTrainer(adapter)
    
    # 模拟数据
    N = 100
    states = torch.randn(N, H)
    bw = F.softmax(torch.randn(N, num_patterns), dim=-1)
    actions = torch.randn(N, num_patterns)
    old_log_probs = torch.randn(N)
    advantages = torch.randn(N)
    returns = torch.randn(N)
    
    stats = trainer.update(states, bw, actions, old_log_probs, advantages, returns)
    print(f"Training stats: {stats}")
    
    # 测试奖励计算
    print("\nTesting Reward Computer:")
    reward_computer = RewardComputer(alpha=0.1)
    
    pred = torch.randn(B, 96, 7)
    target = torch.randn(B, 96, 7)
    input_struct = F.softmax(torch.randn(B, 5), dim=-1)
    output_struct = F.softmax(torch.randn(B, 5), dim=-1)
    
    reward = reward_computer.compute_total_reward(pred, target, input_struct, output_struct)
    print(f"Rewards: {reward.shape}, sample: {reward[:3]}")
