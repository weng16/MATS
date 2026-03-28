"""
创新点 C8: 三阶段渐进式训练 (Three-Stage Progressive Training)

阶段一: 预训练 (Self-Supervised)
- 目标: 让结构编码器学会识别时序结构
- 方法: 对比学习 + 掩码重建 + 结构预测

阶段二: 端到端微调 (Supervised)
- 目标: 在下游任务上优化整个框架
- 方法: 使用完整的联合损失 L_total

阶段三: 强化精调 (RFT)
- 目标: 优化路由策略，进一步提升任务性能
- 方法: PPO算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Callable
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Stage1Pretrainer:
    """
    阶段一: 自监督预训练
    
    目标: 让结构编码器学会识别时序结构
    
    方法:
    1. 对比学习: 相似结构的片段靠近，不同结构的远离
    2. 掩码重建: 掩盖部分时间点，预测缺失值
    3. 结构预测: 预测人工构造的结构标签
    """
    
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        temperature: float = 0.07,
        mask_ratio: float = 0.15,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        structure_pred_weight: float = 0.5
    ):
        self.model = model
        self.temperature = temperature
        self.mask_ratio = mask_ratio

        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        self.structure_pred_weight = structure_pred_weight

        # 只训练编码器 + 权重估计器 + 重建头
        self.optimizer = torch.optim.AdamW(
            list(model.structure_encoder.parameters()) +
            list(model.weight_estimator.parameters()) +
            list(model.reconstruction_head.parameters()),   # ← 直接用 model 的子模块
            lr=lr,
            weight_decay=0.01
        )

        # 学习率调度
        self.scheduler = None
        # reconstruction_head 已注册为 model.reconstruction_head，无需额外创建
    
    def create_augmented_pairs(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建数据增强对
        
        增强方式:
        - 时间裁剪
        - 加噪声
        - 时间翻转
        """
        B, L, D = x.shape
        device = x.device
        
        # 正样本对: 同一序列的不同增强
        x1 = self._augment(x)
        x2 = self._augment(x)
        
        # 标签: 对角线为正样本
        labels = torch.eye(B, device=device)
        
        return x1, x2, labels
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """应用数据增强"""
        B, L, D = x.shape
        
        # 随机选择增强方式
        aug_type = np.random.choice(['noise', 'crop', 'scale'])
        
        if aug_type == 'noise':
            # 加高斯噪声
            noise = torch.randn_like(x) * 0.1
            return x + noise
            
        elif aug_type == 'crop':
            # 随机裁剪后填充
            crop_len = int(L * 0.8)
            start = np.random.randint(0, L - crop_len)
            cropped = x[:, start:start+crop_len, :]
            # 填充回原长度
            padded = F.pad(cropped, (0, 0, 0, L - crop_len))
            return padded
            
        elif aug_type == 'scale':
            # 随机缩放
            scale = 0.8 + 0.4 * torch.rand(B, 1, 1, device=x.device)
            return x * scale
        
        return x
    
    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        对比学习损失 (InfoNCE)
        """
        B = z1.shape[0]
        
        # 归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # 相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature  # [B, B]
        
        # 标签: 对角线为正样本
        labels = torch.arange(B, device=z1.device)
        
        # 双向对比损失
        loss = F.cross_entropy(sim_matrix, labels) + \
               F.cross_entropy(sim_matrix.T, labels)
        
        return loss / 2
    
    def mask_reconstruction_loss(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        掩码重建损失
        """
        B, L, D = x.shape
        device = x.device

        # 生成掩码
        mask = torch.bernoulli(
            torch.full((B, L), 1 - self.mask_ratio, device=device)
        ).unsqueeze(-1)  # [B, L, 1]

        # 应用掩码
        masked_x = x * mask

        # 编码
        z = self.model.structure_encoder(masked_x, return_sequence=True)  # [B, L, H]

        # 重建 — 使用 model 内部的 reconstruction_head（与 model 同设备）
        reconstructed = self.model.reconstruction_head(z)  # [B, L, D]

        # 只计算被掩盖位置的损失
        inv_mask = 1 - mask
        loss = F.mse_loss(reconstructed * inv_mask, x * inv_mask)

        return loss
    
    def structure_prediction_loss(
        self,
        x: torch.Tensor,
        pseudo_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        结构预测损失
        
        pseudo_labels: 基于规则生成的伪标签
        """
        # 编码
        z = self.model.structure_encoder(x)
        
        # 预测权重
        weights = self.model.weight_estimator(z)
        
        # 交叉熵损失
        loss = F.cross_entropy(weights, pseudo_labels)
        
        return loss
    
    def generate_pseudo_labels(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成结构伪标签
        
        基于简单规则检测主导模式
        """
        B, L, D = x.shape
        device = x.device
        
        labels = []
        
        for i in range(B):
            xi = x[i].mean(dim=-1)  # [L]
            
            # 简单规则检测
            # 周期性: 自相关
            acf = self._autocorr(xi)
            
            # 趋势性: 线性拟合斜率
            t = torch.arange(L, dtype=torch.float, device=device)
            slope = ((xi - xi.mean()) * (t - t.mean())).sum() / ((t - t.mean()) ** 2).sum()
            
            # 噪声: 方差
            noise_level = xi.std()
            
            # 判断主导模式
            if acf > 0.5:
                label = 0  # periodic
            elif abs(slope) > 0.1:
                label = 1  # trend
            elif noise_level > 1.0:
                label = 2  # noise
            else:
                label = 4  # general
            
            labels.append(label)
        
        return torch.tensor(labels, device=device)
    
    def _autocorr(self, x: torch.Tensor, lag: int = 7) -> float:
        """计算自相关系数"""
        if len(x) <= lag:
            return 0.0
        
        x_centered = x - x.mean()
        n = len(x) - lag
        
        numerator = (x_centered[:n] * x_centered[lag:]).sum()
        denominator = (x_centered ** 2).sum() + 1e-8
        
        return (numerator / denominator).item()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        contrastive_losses = []
        reconstruction_losses = []
        
        pbar = tqdm(dataloader, desc=f"Stage1 Epoch {epoch}")
        
        for batch in pbar:
            x = batch['x'].to(next(self.model.parameters()).device)
            
            # 对比学习
            x1, x2, _ = self.create_augmented_pairs(x)
            z1 = self.model.structure_encoder(x1)
            z2 = self.model.structure_encoder(x2)
            loss_contrastive = self.contrastive_loss(z1, z2)
            
            # 掩码重建
            loss_reconstruction = self.mask_reconstruction_loss(x)
            
            # 结构预测
            pseudo_labels = self.generate_pseudo_labels(x)
            loss_structure = self.structure_prediction_loss(x, pseudo_labels)
            
            # 总损失
            loss = (
                self.contrastive_weight * loss_contrastive +
                self.reconstruction_weight * loss_reconstruction +
                self.structure_pred_weight * loss_structure
            )
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            contrastive_losses.append(loss_contrastive.item())
            reconstruction_losses.append(loss_reconstruction.item())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'contr': f'{loss_contrastive.item():.4f}',
                'recon': f'{loss_reconstruction.item():.4f}'
            })
        
        return {
            'total_loss': total_loss / len(dataloader),
            'contrastive_loss': np.mean(contrastive_losses),
            'reconstruction_loss': np.mean(reconstruction_losses)
        }


class Stage2Finetuner:
    """
    Stage 2: End-to-end joint fine-tuning with per-module learning rates,
    DAG warmup, gradient clipping, and prototype orthogonality regularization.
    """

    def __init__(
        self,
        model,
        joint_loss,
        lr: float = 5e-4,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        prototype_lr_scale: float = 0.1,
        wcomm_lr_scale: float = 3.0,
    ):
        self.model = model
        self.joint_loss = joint_loss
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_grad_norm = max_grad_norm

        # Per-module parameter groups with different learning rates
        param_groups = self._build_param_groups(
            model, lr, weight_decay, prototype_lr_scale, wcomm_lr_scale
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        self.scheduler = self._create_scheduler(lr)

    @staticmethod
    def _build_param_groups(model, lr, wd, proto_scale, wcomm_scale):
        """Assign different LRs to different modules.

        - Prototype parameters: slower (proto_scale × lr)
        - W_comm: faster (wcomm_scale × lr)
        - Everything else: base lr
        """
        proto_params, wcomm_params, other_params = [], [], []
        proto_names = {'weight_estimator.pattern_prototypes'}
        wcomm_names = {'communication.W_comm', 'communication.W_instant',
                       'communication.W_temporal'}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in proto_names:
                proto_params.append(param)
            elif name in wcomm_names:
                wcomm_params.append(param)
            else:
                other_params.append(param)

        return [
            {'params': other_params, 'lr': lr, 'weight_decay': wd},
            {'params': proto_params, 'lr': lr * proto_scale, 'weight_decay': wd,
             'name': 'prototypes'},
            {'params': wcomm_params, 'lr': lr * wcomm_scale, 'weight_decay': 0.0,
             'name': 'W_comm'},
        ]

    def _create_scheduler(self, lr: float):
        """LR warmup (linear) then cosine annealing."""
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return max(epoch / max(self.warmup_epochs, 1), 1e-2)
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        total_losses = {
            'total': 0, 'task': 0, 'consist': 0,
            'sparse': 0, 'causal': 0, 'balance': 0, 'ortho': 0
        }

        pbar = tqdm(dataloader, desc=f"Stage2 Epoch {epoch}")

        for batch in pbar:
            device = next(self.model.parameters()).device
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            output = self.model(x)
            prediction = output['prediction']

            # Pass prototypes for orthogonality regularization
            prototypes = None
            if hasattr(self.model, 'weight_estimator'):
                prototypes = self.model.weight_estimator.pattern_prototypes

            losses = self.joint_loss(
                prediction=prediction,
                target=y,
                input_structure=output['input_structure'],
                output_structure=output['output_structure'],
                expert_weights=output['expert_weights'],
                W_comm=output['W_comm'],
                router_logits=output.get('router_logits'),
                prototypes=prototypes,
            )

            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Advance DAG warmup counter
            if hasattr(self.joint_loss, 'step'):
                self.joint_loss.step()

            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()

            pbar.set_postfix({
                'loss': f'{losses["total"].item():.4f}',
                'task': f'{losses["task"].item():.4f}',
                'dag':  f'{losses["causal"].item():.4f}',
                'γ':    f'{self.joint_loss.gamma:.4f}',
            })

        self.scheduler.step()

        return {k: v / max(len(dataloader), 1) for k, v in total_losses.items()}
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_mse = 0
        total_mae = 0
        total_consistency = 0
        
        with torch.no_grad():
            for batch in dataloader:
                device = next(self.model.parameters()).device
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                output = self.model(x)
                prediction = output['prediction']
                
                total_mse += F.mse_loss(prediction, y).item()
                total_mae += F.l1_loss(prediction, y).item()
                total_consistency += output['consistency_score'].mean().item()
        
        return {
            'val_mse': total_mse / len(dataloader),
            'val_mae': total_mae / len(dataloader),
            'val_consistency': total_consistency / len(dataloader)
        }


class Stage3RFTTrainer:
    """
    阶段三: 强化学习精调 (RFT)

    目标: 优化路由策略，进一步提升任务性能

    完整 PPO 循环:
    1. forward (use_rft=True) 采集轨迹
    2. RewardComputer 计算奖励 (任务性能 + 结构一致性)
    3. GAE 计算优势函数
    4. PPOTrainer.update 更新策略
    """

    def __init__(
        self,
        model,
        tool_adapter,
        ppo_trainer,
        reward_computer,
        lr: float = 1e-4,
        collect_steps: int = 1,        # 每次 PPO 更新前收集几个 batch 的轨迹
        value_bootstrap: bool = True   # 是否用 value_head 做 bootstrap
    ):
        self.model          = model
        self.tool_adapter   = tool_adapter
        self.ppo_trainer    = ppo_trainer
        self.reward_computer = reward_computer
        self.collect_steps  = collect_steps
        self.value_bootstrap = value_bootstrap

        # 冻结非路由参数
        self._freeze_model()

    def _freeze_model(self):
        """冻结骨干网络，只训练路由相关参数"""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.weight_estimator.parameters():
            param.requires_grad = True
        for param in self.tool_adapter.parameters():
            param.requires_grad = True

    def collect_trajectory(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        收集一条完整轨迹

        返回字段:
            state, base_weights, action (权重调整量),
            reward, log_prob, value
        """
        # 编码结构特征
        with torch.no_grad():
            z = self.model.structure_encoder(x)                # [B, H]
            base_weights = self.model.weight_estimator(z)      # [B, 5]

        # Tool Adapter 采样动作（随机策略，记录 log_prob）
        adjusted_weights, log_prob, value = self.tool_adapter(
            z, base_weights, deterministic=False
        )
        action = adjusted_weights.detach() - base_weights.detach()  # [B, 5]

        # 用调整后权重做预测（不需要梯度，reward 是外部信号）
        with torch.no_grad():
            fused, _ = self.model.expert_fusion(x, adjusted_weights, return_expert_outputs=True)

            # 因果通信
            agent_states = torch.stack(
                [fused.mean(dim=1)] * self.model.num_agents, dim=1
            )
            comm_out = self.model.communication(agent_states).mean(dim=1)
            fused = fused + 0.1 * comm_out.unsqueeze(1)

            # 预测
            task_name = {v: k for k, v in self.model.TASK_TYPES.items()}.get(
                self.model.task_type, 'forecast'
            )
            head = self.model.task_heads[task_name] if task_name in self.model.task_heads else self.model.task_heads['forecast']
            prediction = head(fused)
            if self.model.seq_transform is not None:
                prediction = prediction.transpose(1, 2)
                prediction = self.model.seq_transform(prediction)
                prediction = prediction.transpose(1, 2)

            # 计算输出结构（用于结构一致性奖励）
            output_z        = self.model.structure_encoder(prediction)
            output_structure = self.model.weight_estimator(output_z)

            # 计算奖励
            reward = self.reward_computer.compute_total_reward(
                prediction, y, base_weights, output_structure
            )  # [B]

        return {
            'state':        z.detach(),            # [B, H]
            'base_weights': base_weights.detach(), # [B, 5]
            'action':       action,                # [B, 5]
            'reward':       reward,                # [B]
            'log_prob':     log_prob.detach(),     # [B]
            'value':        value.squeeze(-1).detach(),  # [B]
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个 epoch

        流程:
        1. 遍历 dataloader 收集全部轨迹
        2. 计算 GAE 优势函数
        3. 调用 PPOTrainer.update
        """
        self.model.eval()        # 骨干网络保持 eval（已冻结）
        self.tool_adapter.train()

        device = next(self.model.parameters()).device

        all_states, all_base_weights = [], []
        all_actions, all_rewards    = [], []
        all_log_probs, all_values   = [], []

        pbar = tqdm(dataloader, desc=f"Stage3 RFT Epoch {epoch}")

        for batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            traj = self.collect_trajectory(x, y)

            all_states.append(traj['state'])
            all_base_weights.append(traj['base_weights'])
            all_actions.append(traj['action'])
            all_rewards.append(traj['reward'])
            all_log_probs.append(traj['log_prob'])
            all_values.append(traj['value'])

            pbar.set_postfix({'reward': f"{traj['reward'].mean().item():.4f}"})

        # 拼接所有 batch
        states      = torch.cat(all_states,       dim=0)
        base_weights = torch.cat(all_base_weights, dim=0)
        actions     = torch.cat(all_actions,      dim=0)
        rewards     = torch.cat(all_rewards,      dim=0)
        log_probs   = torch.cat(all_log_probs,    dim=0)
        values      = torch.cat(all_values,       dim=0)

        # ---- GAE 优势估计 ----
        # 时序预测场景：每个 batch 视为独立 episode，done=True
        # 简化 GAE：advantage = reward - value_baseline
        if self.value_bootstrap:
            advantages = rewards - values
        else:
            advantages = rewards - rewards.mean()

        returns = rewards  # 无折扣（每步都是终止状态）

        # ---- PPO 更新 ----
        stats = self.ppo_trainer.update(
            states, base_weights, actions,
            log_probs, advantages, returns
        )

        return {
            'avg_reward':   rewards.mean().item(),
            'avg_advantage': advantages.mean().item(),
            **stats
        }


class ThreeStageTrainer:
    """
    创新点C8: 三阶段渐进式训练

    整合三个训练阶段
    """

    def __init__(
        self,
        model,
        joint_loss,
        config: Dict,
        tool_adapter=None,
        ppo_trainer=None,
        reward_computer=None,
    ):
        from src.models.tool_adapter import ToolAdapterRFT, PPOTrainer, RewardComputer

        self.model  = model
        self.config = config

        # ---- Stage 1: 自监督预训练 ----
        self.stage1 = Stage1Pretrainer(
            model,
            lr=config.get('stage1_lr', 1e-3)
        )

        # ---- Stage 2: 端到端微调 ----
        self.stage2 = Stage2Finetuner(
            model,
            joint_loss,
            lr=config.get('stage2_lr', 5e-4),
            warmup_epochs=config.get('warmup_epochs', 10),
            total_epochs=config.get('stage2_epochs', 100),
            max_grad_norm=config.get('max_grad_norm', 1.0),
            prototype_lr_scale=config.get('prototype_lr_scale', 0.1),
            wcomm_lr_scale=config.get('wcomm_lr_scale', 3.0),
        )

        # ---- Stage 3: RFT ----
        # 若未传入，则自动用 model.tool_adapter 初始化
        if tool_adapter is None:
            if hasattr(model, 'tool_adapter') and model.tool_adapter is not None:
                tool_adapter = model.tool_adapter
            else:
                tool_adapter = ToolAdapterRFT(
                    hidden_dim=model.hidden_dim,
                    num_patterns=model.num_experts
                ).to(next(model.parameters()).device)

        if ppo_trainer is None:
            ppo_trainer = PPOTrainer(
                tool_adapter,
                lr=config.get('stage3_lr', 1e-4),
                clip_epsilon=config.get('ppo_clip', 0.2),
                ppo_epochs=config.get('ppo_epochs', 4),
            )

        if reward_computer is None:
            reward_computer = RewardComputer(
                alpha=config.get('reward_alpha', 0.1)
            )

        self.stage3 = Stage3RFTTrainer(
            model,
            tool_adapter,
            ppo_trainer,
            reward_computer,
            lr=config.get('stage3_lr', 1e-4)
        )

        # 最佳模型状态
        self.best_state  = None
        self.best_metric = float('inf')
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        unlabeled_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        完整的三阶段训练
        """
        history = {
            'stage1': [], 'stage2': [], 'stage3': [],
            'validation': []
        }
        
        # ========== 阶段一: 预训练 ==========
        logger.info("Starting Stage 1: Self-Supervised Pretraining")
        
        stage1_epochs = self.config.get('stage1_epochs', 50)
        data_loader = unlabeled_loader if unlabeled_loader is not None else train_loader
        
        for epoch in range(stage1_epochs):
            metrics = self.stage1.train_epoch(data_loader, epoch)
            history['stage1'].append(metrics)
            
            logger.info(f"Stage1 Epoch {epoch}: {metrics}")
        
        # ========== 阶段二: 端到端微调 ==========
        logger.info("Starting Stage 2: End-to-End Fine-tuning")
        
        stage2_epochs = self.config.get('stage2_epochs', 100)
        
        for epoch in range(stage2_epochs):
            train_metrics = self.stage2.train_epoch(train_loader, epoch)
            val_metrics = self.stage2.validate(val_loader)
            
            history['stage2'].append(train_metrics)
            history['validation'].append(val_metrics)
            
            # 保存最佳模型
            if val_metrics['val_mse'] < self.best_metric:
                self.best_metric = val_metrics['val_mse']
                self.best_state = self.model.state_dict().copy()
            
            logger.info(f"Stage2 Epoch {epoch}: Train {train_metrics}, Val {val_metrics}")
        
        # 恢复最佳模型
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        # ========== 阶段三: RFT ==========
        logger.info("Starting Stage 3: Reinforced Fine-Tuning")
        
        stage3_epochs = self.config.get('stage3_epochs', 20)
        
        for epoch in range(stage3_epochs):
            metrics = self.stage3.train_epoch(train_loader, epoch)
            history['stage3'].append(metrics)
            
            logger.info(f"Stage3 Epoch {epoch}: {metrics}")
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                device = next(self.model.parameters()).device
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                output = self.model(x)
                prediction = output['prediction']
                
                all_predictions.append(prediction.cpu())
                all_targets.append(y.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }


if __name__ == "__main__":
    # 测试代码需要完整的模型定义
    print("Three-stage trainer module loaded successfully.")
    print("Stages:")
    print("  1. Self-Supervised Pretraining")
    print("  2. End-to-End Fine-tuning")
    print("  3. Reinforced Fine-Tuning (RFT)")
