# StructRouter

**结构感知的多智能体混合模式时序分析框架**

**Structure-Aware Multi-Agent Framework for Mixed-Mode Time Series Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-green.svg)](#)

> 📖 可视化方法框架图请查看 [docs/architecture.html](docs/architecture.html)

---

## 概述

时序预测在面对**混合结构模式**（周期性、趋势性、噪声、突变等在不同段落间交替出现）时依然极具挑战性。现有的混合专家（MoE）方法依赖**硬路由（Hard Routing）**——每次仅激活少量专家——导致专家坍塌、信息丢失和可解释性不足。

**StructRouter** 提出了一种全新的思路：将时序数据建模为**可学习结构模式的混合体**，采用**软路由（Soft Routing）**和**可解释因果通信（Causal Communication）** 实现多智能体协作预测。

### 核心创新点

| 编号 | 创新点 | 说明 |
|:----:|--------|------|
| **C1** | 多模式权重估计 | 原型相似度 + MLP 双路径融合，学习可解释的结构原型 |
| **C2** | 软加权专家融合 | 所有专家按权重参与，避免专家坍塌，保留完整知识 |
| **C3** | 工具适配器 (RFT) | 基于 PPO 的强化微调，优化路由策略 |
| **C4** | SCM 因果通信 | 结构因果模型 + NOTEARS DAG 约束，可解释的专家间通信 |
| **C5** | 时序因果扩展 | 支持滞后因果关系的时序 SCM |
| **C6** | 验证智能体 | 一致性 / 有效性 / 不确定性三维检验 + 梯度感知重路由 |
| **C7** | 联合损失函数 | 任务损失 + 结构一致性 + DAG 约束 + 负载均衡 + 正交正则 |
| **C8** | 三阶段渐进训练 | 自监督预训练 → 端到端微调 → 强化微调 |

---

## 方法详解

### 1. 问题定义

给定多变量时序输入 $\mathbf{x} \in \mathbb{R}^{L \times D}$（回看窗口 $L$，$D$ 个变量），目标是预测未来值 $\hat{\mathbf{y}} \in \mathbb{R}^{P \times D}$（预测长度 $P$）。StructRouter 学习将 $\mathbf{x}$ 分解为 $K$ 种结构模式，并路由到专业化的专家网络。

### 2. 可学习结构编码器（Structure Encoder）

采用 **因果时序卷积（CausalConv1d）** + **多头自注意力** 的组合架构：

- **TCN 层**：指数膨胀卷积（$d_l = 2^l$），捕获不同尺度的局部时序模式
- **Self-Attention 层**：捕获全局依赖关系
- **统计特征融合**：均值、标准差、趋势斜率、变化强度、自相关系数

输出全局特征向量 $\mathbf{z}_{\text{global}} \in \mathbb{R}^H$ 和序列特征 $\mathbf{z}_{\text{seq}} \in \mathbb{R}^{L \times H}$。

### 3. 多模式权重估计（C1）

核心公式：

$$\mathbf{w} = \text{Softmax}\left( \frac{\text{MLP}(\mathbf{z}) + \text{softplus}(\lambda) \cdot \text{Sim}(\mathbf{z}, \mathbf{P})}{\tau} \right)$$

其中 $\mathbf{P} \in \mathbb{R}^{K \times H}$ 是正交初始化的模式原型，$\lambda$ 是可学习的平衡参数。通过**原型正交正则化** $\mathcal{L}_{\text{ortho}} = \|\mathbf{P}^\top\mathbf{P} - \mathbf{I}\|_F^2$ 保持原型多样性。

### 4. 软加权专家融合（C2）

不同于硬路由，所有 5 个专家在每次前向传播中均按权重贡献：

$$\text{Output} = \sum_{i=1}^{K} w_i \cdot \text{Expert}_i(\mathbf{x}) + \text{Residual}(\mathbf{x})$$

| 专家 | 职责 | 关键技术 |
|------|------|---------|
| **Periodic Expert** | 周期性模式 | 傅里叶特征 + 周期感知注意力 |
| **Trend Expert** | 趋势性模式 | 多尺度平滑卷积 + BiLSTM |
| **Noise Expert** | 噪声处理 | 鲁棒编码器 + 去噪 Transformer |
| **Abrupt Expert** | 突变点检测 | 差分编码 + 分段注意力 |
| **General Expert** | 通用兜底 | 标准 Transformer 编码器 |

### 5. SCM 因果通信（C4）

专家被视为智能体，通过**结构因果模型**进行消息传递：

$$\mathbf{S}_j^{\text{new}} = \mathbf{S}_j^{\text{old}} + \text{gate} \cdot \text{Receive}\left( \sum_{i} W[i,j] \cdot \text{Message}(\mathbf{S}_i) \right)$$

通信矩阵 $\mathbf{W}_{\text{comm}}$ 通过 **NOTEARS DAG 约束**保证无环性：

$$h(\mathbf{W}) = \text{tr}(\exp(\mathbf{W} \odot \mathbf{W})) - N = 0$$

### 6. 验证智能体与闭环校正（C6）

三维验证机制：
- **一致性检验**：输入/输出结构特征的余弦相似度
- **有效性检验**：统计范围、异常值、偏度检查
- **不确定性检验**：MC Dropout 估计认知/偶然不确定性

验证失败时触发**梯度感知重路由**：计算建议权重并产生二次预测，按置信度混合。

### 7. 联合损失函数（C7）

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{consist}} + \beta \mathcal{L}_{\text{sparse}} + \gamma(t) \mathcal{L}_{\text{causal}} + \delta \mathcal{L}_{\text{balance}} + \varepsilon \mathcal{L}_{\text{ortho}}$$

其中 $\gamma(t)$ 使用**线性预热**，避免 DAG 惩罚在训练早期爆炸。

### 8. 三阶段渐进训练（C8）

| 阶段 | 目标 | 训练内容 |
|------|------|---------|
| Stage 1 | 自监督预训练 | 编码器 + 权重估计器（对比学习、掩码重建、结构预测） |
| Stage 2 | 端到端微调 | 全模型联合训练，分模块学习率，余弦退火 |
| Stage 3 | 强化微调 (RFT) | 冻结骨干，PPO 训练路由器和工具适配器 |

---

## 安装

```bash
conda create -n structrouter python=3.10
conda activate structrouter
pip install -r requirements.txt
```

## 数据集

将 CSV 文件放在 `data/{DatasetName}/` 目录下：

| 数据集 | 变量数 | 频率 | 来源 |
|--------|--------|------|------|
| ETTh1/h2 | 7 | 1h | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTm1/m2 | 7 | 15min | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Weather | 21 | 10min | [Autoformer](https://github.com/thuml/Autoformer) |
| ECL | 321 | 1h | [Autoformer](https://github.com/thuml/Autoformer) |
| Traffic | 862 | 1h | [Autoformer](https://github.com/thuml/Autoformer) |

---

## 训练

```bash
# 单实验 (ETTh1, pred_len=96)
python scripts/train.py --config-name=main/etth1_96

# 指定数据集和预测长度
python scripts/train.py --config-name=main/weather_336

# 收敛性验证
python scripts/test_convergence.py
```

### 配置文件

- **主实验** (`configs/main/`): `{etth1,etth2,ettm1,ettm2,weather,ecl,traffic}_{96,192,336,720}.yaml`
- **消融实验** (`configs/ablation/`): `{no_segment,no_causal,no_verification,...}_*.yaml`

---

## 项目结构

```
MATS/
├── configs/                    # Hydra 配置文件
│   ├── _base_/                 # 基础配置（训练、数据集、模型）
│   ├── main/                   # 主实验配置
│   └── ablation/               # 消融实验配置
├── data/                       # 数据集目录
├── docs/
│   ├── architecture.html       # 📊 交互式方法框架可视化
│   ├── abstract.md             # 论文摘要
│   └── convergence_diagnosis.md
├── paper/                      # ACM MM 2026 论文源文件
│   ├── main.tex
│   └── contents/               # 各章节 .tex 文件
├── scripts/
│   ├── train.py                # 训练入口（Hydra + BasicTS）
│   ├── visualize_training.py   # 训练过程可视化
│   └── todo_exp.sh             # 全量实验脚本
├── src/
│   ├── models/
│   │   ├── struct_router.py       # 🧠 主模型（调度所有组件）
│   │   ├── structure_encoder.py   # 结构编码器 (TCN + Attention)
│   │   ├── weight_estimator.py    # C1: 多模式权重估计
│   │   ├── expert_fusion.py       # C2: 软加权专家融合
│   │   ├── experts.py             # 5 个专家网络
│   │   ├── causal_communication.py# C4: SCM 因果通信
│   │   ├── verification_agent.py  # C6: 验证智能体
│   │   ├── tool_adapter.py        # C3: PPO 强化微调
│   │   ├── segment_processor.py   # 分段检测与融合
│   │   └── revin.py               # RevIN 归一化
│   ├── losses/
│   │   └── joint_loss.py          # C7: 联合损失
│   ├── trainers/
│   │   └── three_stage_trainer.py # C8: 三阶段训练器
│   └── basicts_adapter/
│       └── mats_arch.py           # BasicTS 适配器
└── requirements.txt
```

---

## 引用

```
[论文接收后更新]
```

## License

MIT
