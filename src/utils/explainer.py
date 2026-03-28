"""
可解释性工具

提供模型预测的解释:
- 因果图可视化
- 专家贡献度分析
- 结构特征分析
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class CausalExplainer:
    """
    因果可解释性分析器
    
    提供:
    1. 专家贡献度分析 (Shapley Value近似)
    2. 因果路径追踪
    3. 结构一致性分析
    """
    
    def __init__(self, model):
        self.model = model
        self.expert_names = ['periodic', 'trend', 'noise', 'changepoint', 'general']
    
    def explain_prediction(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        生成预测解释
        
        参数:
            x: [B, L, D] 输入序列
            y: [B, L_pred, D] 目标序列 (可选，用于计算贡献度)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            output = self.model(x, return_details=True)
        
        # 1. 专家权重分析
        expert_weights = output['expert_weights']
        weight_analysis = self._analyze_weights(expert_weights)
        
        # 2. 因果图分析
        causal_analysis = self._analyze_causal_graph()
        
        # 3. 结构一致性分析
        structure_analysis = {
            'consistency_score': output['consistency_score'].cpu().numpy(),
            'input_structure': output['input_structure'].cpu().numpy(),
            'output_structure': output['output_structure'].cpu().numpy()
        }
        
        # 4. 如果有目标，计算专家贡献度
        if y is not None:
            y = y.to(device)
            contributions = self._compute_contributions(x, y)
        else:
            contributions = None
        
        return {
            'prediction': output['prediction'].cpu().numpy(),
            'expert_weights': expert_weights.cpu().numpy(),
            'weight_analysis': weight_analysis,
            'causal_analysis': causal_analysis,
            'structure_analysis': structure_analysis,
            'contributions': contributions
        }
    
    def _analyze_weights(self, weights: torch.Tensor) -> Dict:
        """分析专家权重分布"""
        weights_np = weights.cpu().numpy()
        
        # 平均权重
        avg_weights = weights_np.mean(axis=0)
        
        # 主导专家
        dominant_expert_idx = avg_weights.argmax()
        dominant_expert = self.expert_names[dominant_expert_idx]
        
        # 权重熵 (衡量分布均匀度)
        entropy = -(weights_np * np.log(weights_np + 1e-8)).sum(axis=-1).mean()
        max_entropy = np.log(len(self.expert_names))
        uniformity = entropy / max_entropy
        
        return {
            'average_weights': {name: avg_weights[i] for i, name in enumerate(self.expert_names)},
            'dominant_expert': dominant_expert,
            'uniformity': uniformity,
            'entropy': entropy
        }
    
    def _analyze_causal_graph(self) -> Dict:
        """分析因果图"""
        causal_graph = self.model.get_causal_graph()
        
        # 计算每个节点的入度和出度
        adjacency = causal_graph['adjacency']
        in_degrees = adjacency.sum(axis=0)
        out_degrees = adjacency.sum(axis=1)
        
        # 找到关键节点 (入度或出度最高)
        key_receiver = self.expert_names[in_degrees.argmax()]
        key_sender = self.expert_names[out_degrees.argmax()]
        
        # 边强度排序
        edges_sorted = sorted(causal_graph['edges'], 
                             key=lambda e: abs(e['weight']), reverse=True)
        
        top_edges = []
        for edge in edges_sorted[:3]:
            top_edges.append({
                'from': self.expert_names[edge['from']],
                'to': self.expert_names[edge['to']],
                'weight': edge['weight']
            })
        
        return {
            'num_edges': len(causal_graph['edges']),
            'key_receiver': key_receiver,
            'key_sender': key_sender,
            'top_edges': top_edges,
            'adjacency': adjacency
        }
    
    def _compute_contributions(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算专家贡献度 (Leave-One-Out近似)
        """
        device = x.device
        contributions = {}
        
        # 完整预测的损失
        with torch.no_grad():
            output = self.model(x)
            full_loss = F.mse_loss(output['prediction'], y).item()
        
        # 逐个关闭专家
        for i, name in enumerate(self.expert_names):
            with torch.no_grad():
                z = self.model.structure_encoder(x)
                weights = self.model.weight_estimator(z)
                
                # 关闭第i个专家
                masked_weights = weights.clone()
                masked_weights[:, i] = 0
                masked_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-8)
                
                # 重新预测
                prediction = self.model._forward_with_weights(x, masked_weights)['prediction']
                partial_loss = F.mse_loss(prediction, y).item()
            
            # 贡献度 = 关闭后损失增加量 (正值表示有正贡献)
            contributions[name] = partial_loss - full_loss
        
        return contributions
    
    def generate_text_explanation(self, explanation: Dict) -> str:
        """
        生成自然语言解释
        """
        lines = []
        
        # 权重分析
        wa = explanation['weight_analysis']
        lines.append(f"主导专家: {wa['dominant_expert']} "
                    f"(权重: {wa['average_weights'][wa['dominant_expert']]:.3f})")
        lines.append(f"权重分布均匀度: {wa['uniformity']:.2f}")
        
        # 因果分析
        ca = explanation['causal_analysis']
        lines.append(f"\n因果图分析:")
        lines.append(f"  - 关键信息接收者: {ca['key_receiver']}")
        lines.append(f"  - 关键信息发送者: {ca['key_sender']}")
        lines.append(f"  - 主要因果路径:")
        for edge in ca['top_edges']:
            direction = "→" if edge['weight'] > 0 else "⇢"
            lines.append(f"    {edge['from']} {direction} {edge['to']} "
                        f"(强度: {abs(edge['weight']):.3f})")
        
        # 结构一致性
        sa = explanation['structure_analysis']
        avg_consistency = sa['consistency_score'].mean()
        lines.append(f"\n结构一致性: {avg_consistency:.3f}")
        
        # 贡献度
        if explanation['contributions'] is not None:
            lines.append(f"\n专家贡献度 (基于Leave-One-Out):")
            sorted_contrib = sorted(explanation['contributions'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for name, contrib in sorted_contrib:
                sign = "+" if contrib > 0 else ""
                lines.append(f"  - {name}: {sign}{contrib:.4f}")
        
        return "\n".join(lines)


def visualize_causal_graph(
    model,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    可视化因果图
    """
    try:
        import networkx as nx
    except ImportError:
        print("Please install networkx: pip install networkx")
        return
    
    causal_graph = model.get_causal_graph()
    expert_names = ['periodic', 'trend', 'noise', 'changepoint', 'general']
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    for name in expert_names:
        G.add_node(name)
    
    # 添加边
    for edge in causal_graph['edges']:
        G.add_edge(
            expert_names[edge['from']],
            expert_names[edge['to']],
            weight=edge['weight']
        )
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    
    pos = nx.circular_layout(G)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    # 绘制边
    edges = G.edges(data=True)
    weights = [d['weight'] for (u, v, d) in edges]
    
    # 正权重用蓝色，负权重用红色
    edge_colors = ['blue' if w > 0 else 'red' for w in weights]
    edge_widths = [abs(w) * 3 for w in weights]
    
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors,
        width=edge_widths, alpha=0.7,
        connectionstyle="arc3,rad=0.1",
        ax=ax
    )
    
    ax.set_title("Expert Causal Communication Graph")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return fig


def visualize_expert_weights(
    weights: np.ndarray,
    expert_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    可视化专家权重分布
    
    参数:
        weights: [B, K] 或 [K] 权重数组
    """
    if expert_names is None:
        expert_names = ['Periodic', 'Trend', 'Noise', 'ChangePoint', 'General']
    
    if weights.ndim == 2:
        avg_weights = weights.mean(axis=0)
        std_weights = weights.std(axis=0)
    else:
        avg_weights = weights
        std_weights = None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(expert_names))
    
    bars = ax.bar(x, avg_weights, color='steelblue', alpha=0.8)
    
    if std_weights is not None:
        ax.errorbar(x, avg_weights, yerr=std_weights, fmt='none', 
                   color='black', capsize=5)
    
    # 添加数值标签
    for bar, weight in zip(bars, avg_weights):
        height = bar.get_height()
        ax.annotate(f'{weight:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    ax.set_xticks(x)
    ax.set_xticklabels(expert_names, rotation=45, ha='right')
    ax.set_ylabel('Weight')
    ax.set_title('Expert Weight Distribution')
    ax.set_ylim(0, max(avg_weights) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    print("CausalExplainer module loaded successfully.")
    print("\nUsage:")
    print("  explainer = CausalExplainer(model)")
    print("  explanation = explainer.explain_prediction(x)")
    print("  print(explainer.generate_text_explanation(explanation))")
