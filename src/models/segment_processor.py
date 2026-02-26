"""
Segment-level 时序分割处理模块

核心功能:
1. 将长时序分割成多个segment
2. 为每个segment独立估计模式权重
3. 生成 Segment-level Structure Timeline Output
4. Pattern Abstraction 语义模式抽象

对应图中:
- [0-150: P:0.7|T:0.2|N:0.1] → [150-210: A:0.5|N:0.3|T:0.2] → ...
- Pattern Abstraction (Periodic Seg, Trend Seg, Abrupt Change Seg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange, repeat
import numpy as np


class SegmentDetector(nn.Module):
    """
    自适应时序分段检测器
    
    检测时序中的结构变化点，自动确定分段边界
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_segments: int = 10,
        min_segment_len: int = 16
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_segments = max_segments
        self.min_segment_len = min_segment_len
        
        # 变化点检测网络
        self.change_detector = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()  # 输出变化点概率
        )
        
        # 分段数量预测
        self.num_segments_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, max_segments)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        z_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        检测分段边界
        
        参数:
            x: [B, L, D] 原始时序
            z_seq: [B, L, H] 编码后的序列特征
            
        返回:
            change_probs: [B, L] 每个时间点是变化点的概率
            segment_boundaries: List of [num_segments+1] 边界索引
            num_segments: [B] 预测的分段数
        """
        B, L, H = z_seq.shape
        
        # 检测变化点概率
        z_conv = rearrange(z_seq, 'b l h -> b h l')
        change_probs = self.change_detector(z_conv).squeeze(1)  # [B, L]
        
        # 预测分段数量
        z_global = z_seq.mean(dim=1)  # [B, H]
        num_logits = self.num_segments_predictor(z_global)  # [B, max_segments]
        num_segments = num_logits.argmax(dim=-1) + 1  # [B], 最少1段
        
        return {
            'change_probs': change_probs,
            'num_segments': num_segments,
            'num_logits': num_logits
        }
    
    def get_segment_boundaries(
        self,
        change_probs: torch.Tensor,
        num_segments: torch.Tensor,
        seq_len: int
    ) -> List[List[Tuple[int, int]]]:
        """
        根据变化点概率获取分段边界
        
        返回: List of segments, 每个segment是 (start, end) tuple
        """
        B = change_probs.shape[0]
        all_segments = []
        
        for b in range(B):
            probs = change_probs[b].detach().cpu().numpy()
            n_seg = num_segments[b].item()
            
            # 找到top-k变化点
            if n_seg > 1:
                # 避免边界附近
                probs[:self.min_segment_len] = 0
                probs[-self.min_segment_len:] = 0
                
                # 找峰值
                boundaries = [0]
                for _ in range(n_seg - 1):
                    if probs.max() > 0.3:  # 阈值
                        peak_idx = probs.argmax()
                        boundaries.append(int(peak_idx))
                        # 抑制峰值附近
                        start = max(0, peak_idx - self.min_segment_len)
                        end = min(seq_len, peak_idx + self.min_segment_len)
                        probs[start:end] = 0
                    else:
                        break
                
                boundaries.append(seq_len)
                boundaries = sorted(boundaries)
            else:
                boundaries = [0, seq_len]
            
            # 转换为 (start, end) 对
            segments = [(boundaries[i], boundaries[i+1]) 
                       for i in range(len(boundaries)-1)]
            all_segments.append(segments)
        
        return all_segments


class SegmentLevelWeightEstimator(nn.Module):
    """
    Segment-level 权重估计器
    
    为每个segment独立估计模式权重，输出类似:
    [0-150: P:0.7|T:0.2|N:0.1] → [150-210: A:0.5|N:0.3|T:0.2] → ...
    """
    
    PATTERN_NAMES = ['periodic', 'trend', 'noise', 'breakpoint', 'missing']
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_patterns: int = 5
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        
        # Segment编码器
        self.segment_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 模式原型
        self.pattern_prototypes = nn.Parameter(
            torch.randn(num_patterns, hidden_dim)
        )
        nn.init.orthogonal_(self.pattern_prototypes)
        
        # Segment权重MLP
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_patterns)
        )
    
    def encode_segment(
        self,
        z_seq: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        编码单个segment
        
        参数:
            z_seq: [B, L, H] 完整序列特征
            start, end: segment边界
            
        返回:
            segment_feat: [B, H] segment特征
        """
        segment = z_seq[:, start:end, :]  # [B, seg_len, H]
        
        # LSTM编码
        _, (h_n, _) = self.segment_encoder(segment)
        
        # 合并双向hidden state
        segment_feat = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, H]
        
        return segment_feat
    
    def estimate_segment_weight(
        self,
        segment_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        估计单个segment的模式权重
        
        返回:
            weights: [B, num_patterns] 模式权重
        """
        # MLP预测
        mlp_logits = self.weight_mlp(segment_feat)  # [B, 5]
        
        # 原型相似度
        proto_norm = F.normalize(self.pattern_prototypes, dim=-1)
        feat_norm = F.normalize(segment_feat, dim=-1)
        sim_logits = torch.matmul(feat_norm, proto_norm.T)  # [B, 5]
        
        # 融合
        combined = mlp_logits + 0.5 * sim_logits
        weights = F.softmax(combined, dim=-1)
        
        return weights
    
    def forward(
        self,
        z_seq: torch.Tensor,
        segments: List[List[Tuple[int, int]]]
    ) -> Dict:
        """
        为所有segments估计权重
        
        参数:
            z_seq: [B, L, H] 序列特征
            segments: 每个batch的分段列表
            
        返回:
            segment_weights: List[Tensor] 每个batch每个segment的权重
            segment_profiles: List[Dict] 语义模式描述
        """
        B = z_seq.shape[0]
        
        all_segment_weights = []
        all_segment_profiles = []
        
        for b in range(B):
            batch_weights = []
            batch_profiles = []
            
            for start, end in segments[b]:
                # 编码segment
                seg_feat = self.encode_segment(
                    z_seq[b:b+1], start, end
                )
                
                # 估计权重
                weights = self.estimate_segment_weight(seg_feat)  # [1, 5]
                batch_weights.append(weights.squeeze(0))
                
                # 生成语义描述
                profile = self._generate_profile(
                    weights.squeeze(0), start, end
                )
                batch_profiles.append(profile)
            
            all_segment_weights.append(torch.stack(batch_weights))
            all_segment_profiles.append(batch_profiles)
        
        return {
            'segment_weights': all_segment_weights,  # List of [num_seg, 5]
            'segment_profiles': all_segment_profiles
        }
    
    def _generate_profile(
        self,
        weights: torch.Tensor,
        start: int,
        end: int
    ) -> Dict:
        """
        生成segment的语义模式描述
        
        类似: "[0-150: P:0.7|T:0.2|N:0.1]"
        """
        w = weights.detach().cpu().numpy()
        
        # 找主导模式
        dominant_idx = w.argmax()
        dominant_pattern = self.PATTERN_NAMES[dominant_idx]
        
        # 生成描述字符串
        desc_parts = []
        for i, name in enumerate(self.PATTERN_NAMES):
            if w[i] > 0.1:  # 只显示权重>0.1的模式
                abbr = name[0].upper()  # 首字母缩写
                desc_parts.append(f"{abbr}:{w[i]:.1f}")
        
        profile_str = f"[{start}-{end}: {'|'.join(desc_parts)}]"
        
        return {
            'range': (start, end),
            'weights': w,
            'dominant': dominant_pattern,
            'description': profile_str
        }


class PatternAbstraction(nn.Module):
    """
    Pattern Abstraction 模块
    
    对应图中的:
    - Global Strong Periodic
    - Global Segment Structure  
    - Global Noise Dominant
    - Global Missing Driven
    
    将segment-level模式聚合为全局语义标签
    """
    
    GLOBAL_PATTERNS = [
        'Global Strong Periodic',
        'Global Segment Structure',
        'Global Noise Dominant',
        'Global Missing Driven',
        'Global Trend Dominant'
    ]
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 全局模式分类器
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim // 2),  # H + 5种模式权重
            nn.GELU(),
            nn.Linear(hidden_dim // 2, len(self.GLOBAL_PATTERNS))
        )
    
    def forward(
        self,
        z_global: torch.Tensor,
        segment_weights: List[torch.Tensor]
    ) -> Dict:
        """
        生成全局模式抽象
        
        参数:
            z_global: [B, H] 全局特征
            segment_weights: List of [num_seg, 5] 每个batch的segment权重
            
        返回:
            global_pattern: [B] 全局模式索引
            global_pattern_probs: [B, num_global_patterns] 各模式概率
            semantic_label: List[str] 语义标签
        """
        B = z_global.shape[0]
        
        # 聚合segment权重 (加权平均)
        aggregated_weights = []
        for sw in segment_weights:
            # 使用segment长度作为权重? 这里简化为平均
            avg_w = sw.mean(dim=0)  # [5]
            aggregated_weights.append(avg_w)
        
        agg_weights = torch.stack(aggregated_weights)  # [B, 5]
        
        # 拼接全局特征和聚合权重
        combined = torch.cat([z_global, agg_weights], dim=-1)  # [B, H+5]
        
        # 预测全局模式
        logits = self.global_classifier(combined)  # [B, num_patterns]
        probs = F.softmax(logits, dim=-1)
        
        global_pattern = probs.argmax(dim=-1)  # [B]
        
        # 生成语义标签
        semantic_labels = [self.GLOBAL_PATTERNS[idx.item()] for idx in global_pattern]
        
        return {
            'global_pattern': global_pattern,
            'global_pattern_probs': probs,
            'semantic_labels': semantic_labels,
            'aggregated_weights': agg_weights
        }


class MultiSegmentRouteComposition(nn.Module):
    """
    Multi-Segment Route Composition
    
    对应图中 Cognitive Router 的:
    - Structure-driven Routing (Feasibility Check)
    - Task-driven Selection (Constraint Matching)
    - Multi-Segment Route Composition
    """
    
    TASK_TYPES = ['forecast', 'imputation', 'classification', 'anomaly_detection']
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_patterns: int = 5,
        num_tasks: int = 4
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        self.num_tasks = num_tasks
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)
        
        # Structure-driven 可行性检查
        self.feasibility_checker = nn.Sequential(
            nn.Linear(hidden_dim + num_patterns, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_patterns),
            nn.Sigmoid()  # 每个专家的可行性分数
        )
        
        # Task-driven 约束匹配
        self.task_constraint_matcher = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_patterns)
        )
        
        # 多段路由组合
        self.segment_combiner = nn.Sequential(
            nn.Linear(num_patterns * 2, num_patterns),
            nn.GELU(),
            nn.Linear(num_patterns, num_patterns)
        )
        
        # 可学习的任务-专家亲和矩阵
        # 不同任务对不同专家有不同偏好
        self.task_expert_affinity = nn.Parameter(
            torch.zeros(num_tasks, num_patterns)
        )
        nn.init.xavier_uniform_(self.task_expert_affinity)
    
    def structure_driven_routing(
        self,
        z: torch.Tensor,
        segment_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Structure-driven Routing (Feasibility Check)
        
        根据结构特征检查每个专家的可行性
        """
        # 拼接特征和权重
        combined = torch.cat([z, segment_weights], dim=-1)
        
        # 可行性分数
        feasibility = self.feasibility_checker(combined)  # [B, 5]
        
        return feasibility
    
    def task_driven_selection(
        self,
        z: torch.Tensor,
        task_type: int
    ) -> torch.Tensor:
        """
        Task-driven Selection (Constraint Matching)
        
        根据任务类型调整路由
        """
        B = z.shape[0]
        
        # 获取任务embedding
        task_idx = torch.full((B,), task_type, dtype=torch.long, device=z.device)
        task_emb = self.task_embedding(task_idx)  # [B, H]
        
        # 任务约束匹配
        combined = torch.cat([z, task_emb], dim=-1)
        task_weights = self.task_constraint_matcher(combined)  # [B, 5]
        
        # 加上任务-专家亲和性
        affinity = self.task_expert_affinity[task_type]  # [5]
        task_weights = task_weights + affinity.unsqueeze(0)
        
        return task_weights
    
    def compose_segment_routes(
        self,
        segment_weights: List[torch.Tensor],
        feasibility_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        组合多个segment的路由
        
        返回: [B, num_patterns] 组合后的路由权重
        """
        B = feasibility_scores.shape[0]
        
        composed_weights = []
        for b in range(B):
            sw = segment_weights[b]  # [num_seg, 5]
            
            # 加权平均 (可以用更复杂的策略)
            # 权重与可行性分数成正比
            feas = feasibility_scores[b:b+1].expand(sw.shape[0], -1)  # [num_seg, 5]
            weighted = sw * feas
            composed = weighted.sum(dim=0) / (feas.sum(dim=0) + 1e-8)  # [5]
            
            composed_weights.append(composed)
        
        return torch.stack(composed_weights)  # [B, 5]
    
    def forward(
        self,
        z: torch.Tensor,
        segment_weights: List[torch.Tensor],
        task_type: int = 0
    ) -> Dict:
        """
        完整的多段路由组合
        
        参数:
            z: [B, H] 全局特征
            segment_weights: List of [num_seg, 5]
            task_type: 任务类型 (0-3)
        """
        B = z.shape[0]
        
        # 获取每个batch的聚合权重
        agg_weights = torch.stack([sw.mean(dim=0) for sw in segment_weights])  # [B, 5]
        
        # 1. Structure-driven routing
        feasibility = self.structure_driven_routing(z, agg_weights)
        
        # 2. Task-driven selection
        task_weights = self.task_driven_selection(z, task_type)
        
        # 3. 组合多段路由
        composed_weights = self.compose_segment_routes(segment_weights, feasibility)
        
        # 4. 最终融合
        combined = torch.cat([composed_weights, task_weights], dim=-1)  # [B, 10]
        final_weights = self.segment_combiner(combined)  # [B, 5]
        
        # 应用可行性掩码并归一化
        final_weights = final_weights * feasibility
        final_weights = F.softmax(final_weights, dim=-1)
        
        return {
            'final_weights': final_weights,
            'feasibility_scores': feasibility,
            'task_weights': F.softmax(task_weights, dim=-1),
            'composed_weights': F.softmax(composed_weights, dim=-1)
        }


class SegmentFusion(nn.Module):
    """
    Segment Fusion (Boundary Smoothing)
    
    对应图中右侧的 Segment Fusion 模块
    将多个segment的预测结果平滑融合
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        smoothing_kernel_size: int = 5
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.smoothing_kernel_size = smoothing_kernel_size
        
        # 边界平滑卷积
        self.boundary_smoother = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=smoothing_kernel_size,
            padding=smoothing_kernel_size // 2,
            groups=hidden_dim  # depthwise
        )
        
        # 权重融合网络
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 可学习的边界衰减
        self.boundary_decay = nn.Parameter(torch.tensor(0.1))
    
    def create_boundary_mask(
        self,
        seq_len: int,
        segments: List[Tuple[int, int]],
        device: torch.device
    ) -> torch.Tensor:
        """
        创建边界掩码，边界处权重较低
        """
        mask = torch.ones(seq_len, device=device)
        
        for start, end in segments:
            # 边界区域衰减
            boundary_width = max(3, (end - start) // 10)
            
            # 开始边界
            for i in range(boundary_width):
                if start + i < seq_len:
                    mask[start + i] *= (i + 1) / (boundary_width + 1)
            
            # 结束边界
            for i in range(boundary_width):
                if end - 1 - i >= 0:
                    mask[end - 1 - i] *= (i + 1) / (boundary_width + 1)
        
        return mask
    
    def forward(
        self,
        segment_outputs: List[torch.Tensor],
        segments: List[Tuple[int, int]],
        seq_len: int
    ) -> torch.Tensor:
        """
        融合多个segment的输出
        
        参数:
            segment_outputs: List of [seg_len, H] 各segment输出
            segments: 分段边界 [(start, end), ...]
            seq_len: 原始序列长度
            
        返回:
            fused: [seq_len, H] 融合后的输出
        """
        device = segment_outputs[0].device
        H = segment_outputs[0].shape[-1]
        
        # 初始化输出
        fused = torch.zeros(seq_len, H, device=device)
        weights = torch.zeros(seq_len, device=device)
        
        # 累加各segment贡献
        for seg_out, (start, end) in zip(segment_outputs, segments):
            seg_len = end - start
            
            # 创建边界掩码
            mask = self.create_boundary_mask(seg_len, [(0, seg_len)], device)
            
            # 加权累加
            fused[start:end] += seg_out * mask.unsqueeze(-1)
            weights[start:end] += mask
        
        # 归一化
        weights = weights.clamp(min=1e-8)
        fused = fused / weights.unsqueeze(-1)
        
        # 边界平滑
        fused = rearrange(fused, 'l h -> 1 h l')
        fused = self.boundary_smoother(fused)
        fused = rearrange(fused, '1 h l -> l h')
        
        return fused


if __name__ == "__main__":
    # 测试代码
    B, L, H = 2, 200, 256
    
    print("Testing Segment Processor...")
    
    # 模拟数据
    z_seq = torch.randn(B, L, H)
    z_global = z_seq.mean(dim=1)
    
    # 测试分段检测
    detector = SegmentDetector(hidden_dim=H)
    detect_result = detector(None, z_seq)
    print(f"Change probs shape: {detect_result['change_probs'].shape}")
    print(f"Num segments: {detect_result['num_segments']}")
    
    # 获取分段边界
    segments = detector.get_segment_boundaries(
        detect_result['change_probs'],
        detect_result['num_segments'],
        L
    )
    print(f"Segments: {segments}")
    
    # 测试Segment权重估计
    seg_estimator = SegmentLevelWeightEstimator(hidden_dim=H)
    seg_result = seg_estimator(z_seq, segments)
    print(f"\nSegment weights shapes: {[sw.shape for sw in seg_result['segment_weights']]}")
    print(f"Segment profiles: {seg_result['segment_profiles'][0]}")
    
    # 测试Pattern Abstraction
    abstraction = PatternAbstraction(hidden_dim=H)
    abs_result = abstraction(z_global, seg_result['segment_weights'])
    print(f"\nGlobal pattern: {abs_result['semantic_labels']}")
    
    # 测试Multi-Segment Route Composition
    composer = MultiSegmentRouteComposition(hidden_dim=H)
    route_result = composer(z_global, seg_result['segment_weights'], task_type=0)
    print(f"\nFinal routing weights: {route_result['final_weights']}")
    print(f"Feasibility scores: {route_result['feasibility_scores']}")
