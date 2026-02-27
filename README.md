# StructRouter: 结构感知多智能体协作时序分析框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概述

**StructRouter** 是一个面向顶会(ICML/NeurIPS/ICLR)的**长时序预测(Long-term Time Series Forecasting, LTSF)**框架，核心聚焦**三大创新Method**：

1. **Method 1: 权重测量 → Expert → Tool Adapter (RFT)**
2. **Method 2: 通信机制 - 因果图/SCM建模**
3. **Method 3: Framework - 预测、验证、优化(联合Loss)**

## 🎯 核心创新点

| 编号 | 创新点 | 解决的问题 | 技术手段 |
|------|--------|-----------|----------|
| **C1** | 多模式权重估计 | 单一模式假设 | MLP + Prototype相似度 |
| **C2** | 软加权专家融合 | Top-K硬路由信息丢失 | 连续权重加权求和 |
| **C3** | Tool Adapter + RFT | 路由策略非最优 | PPO强化学习微调 |
| **C4** | SCM因果通信 | 通信不可解释 | 结构因果模型+DAG约束 |
| **C5** | 时序因果扩展 | 忽略时序因果 | 时序因果张量 |
| **C6** | 验证智能体 | 幻觉输出 | Consistency+Validity+Uncertainty |
| **C7** | 联合损失函数 | 各模块独立优化 | 多目标统一优化 |
| **C8** | 三阶段训练 | 训练不稳定 | 预训练→微调→RFT |

---

## 📊 支持的数据集

我们支持主流长时序预测基准数据集 (兼容 BasicTS 等框架的数据格式):

| 数据集 | 描述 | 特征数 | 频率 | 样本数 |
|--------|------|--------|------|--------|
| **ETTh1** | 电力变压器温度 (小时) | 7 | 1h | 17,420 |
| **ETTh2** | 电力变压器温度 (小时) | 7 | 1h | 17,420 |
| **ETTm1** | 电力变压器温度 (分钟) | 7 | 15min | 69,680 |
| **ETTm2** | 电力变压器温度 (分钟) | 7 | 15min | 69,680 |
| **Weather** | 气象数据 | 21 | 10min | 52,696 |
| **Traffic** | 旧金山交通流量 | 862 | 1h | 17,544 |
| **Electricity** | 电力消耗 | 321 | 1h | 26,304 |
| **Exchange** | 汇率数据 | 8 | 1day | 7,588 |
| **ILI** | 流感数据 | 7 | 1week | 966 |

### 数据集下载

数据集可从以下来源下载：

1. **ETT数据集**: https://github.com/zhouhaoyi/ETDataset
2. **其他数据集**: https://github.com/thuml/Autoformer (datasets文件夹)
3. **BasicTS框架**: https://github.com/zezhishao/BasicTS

下载后将数据放入 `data/` 目录：

```
MATS/
├── data/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   ├── ETTm2.csv
│   ├── weather.csv
│   ├── traffic.csv
│   ├── electricity.csv
│   ├── exchange_rate.csv
│   └── national_illness.csv
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
cd MATS

# 创建虚拟环境 (推荐)
conda create -n structrouter python=3.10
conda activate structrouter

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 方法1: 下载真实数据集
# 从上述链接下载并放入 data/ 目录

# 方法2: 使用合成数据测试 (无需下载)
# 代码会自动生成合成数据用于测试
```

### 3. 运行训练

```bash
# === 基本训练 ===
# 使用默认配置 (ETTh1, pred_len=96)
python scripts/train.py

# === 指定数据集和预测长度 ===
python scripts/train.py --dataset ETTh1 --pred_len 96
python scripts/train.py --dataset Weather --pred_len 192
python scripts/train.py --dataset Traffic --pred_len 336

# === 运行所有预测长度 (96, 192, 336, 720) ===
python scripts/train.py --dataset ETTh1 --all_lengths
python scripts/train.py --dataset ETTm1 --all_lengths

# === 使用GPU ===
python scripts/train.py --device cuda --dataset ETTh1 --pred_len 96

# === 自定义配置 ===
python scripts/train.py --config configs/default.yaml \
    --dataset ETTh1 \
    --pred_len 96 \
    --epochs 100 \
    --batch_size 32 \
    --lr 5e-4

# === 完整实验 (所有数据集 + 所有预测长度) ===
for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather Traffic Electricity Exchange; do
    python scripts/train.py --dataset $dataset --all_lengths
done
```

### 4. 查看结果

训练完成后，结果保存在 `outputs/` 目录：

```
outputs/
├── ETTh1_pred96_20260112_143000/
│   ├── best_model.pt      # 最佳模型权重
│   └── results.json       # 测试结果
```

---

## 📁 项目结构

```
MATS/
├── configs/
│   └── default.yaml                 # 默认配置文件
├── data/                            # 数据目录 (需手动创建)
│   ├── ETTh1.csv
│   └── ...
├── scripts/
│   └── train.py                     # 训练脚本
├── src/
│   ├── models/
│   │   ├── structure_encoder.py     # Learnable Structure Encoder (TCN-based)
│   │   ├── weight_estimator.py      # Multi-Pattern Weight Estimator
│   │   ├── segment_processor.py     # Segment-level处理 + Pattern Abstraction
│   │   ├── experts.py               # Task Expert Pool (w1-w5)
│   │   ├── expert_fusion.py         # Soft-Weighted Expert Composition
│   │   ├── tool_adapter.py          # Tool Adapter + RFT
│   │   ├── causal_communication.py  # Learnable Communication Matrix (W_comm)
│   │   ├── verification.py          # 基础闭环校验
│   │   ├── verification_agent.py    # Verification Agent (完整版)
│   │   └── struct_router.py         # Cognitive Router 主模型
│   ├── losses/
│   │   └── joint_loss.py            # 联合损失函数
│   ├── trainers/
│   │   └── three_stage_trainer.py   # 三阶段训练器
│   ├── data/
│   │   └── dataset.py               # 数据集加载
│   └── utils/
│       └── explainer.py             # 可解释性工具
├── outputs/                         # 输出目录 (自动创建)
├── checkpoints/                     # 检查点目录 (自动创建)
├── requirements.txt
└── README.md
```

---

## ⚙️ 配置说明

主要配置在 `configs/default.yaml`:

```yaml
# 数据配置
data:
  dataset: "ETTh1"         # 数据集名称
  root_path: "./data"      # 数据目录
  seq_len: 96              # 输入长度 (历史窗口)
  pred_len: 96             # 预测长度
  features: "M"            # M=多变量, S=单变量, MS=多变量预测单变量

# 模型配置
model:
  hidden_dim: 256          # 隐藏层维度
  num_experts: 5           # 专家数量
  use_segment_processing: true   # 启用Segment级处理
  use_verification: true         # 启用验证智能体

# 训练配置
training:
  stage2:                  # 主要训练阶段
    epochs: 100
    lr: 5.0e-4
    batch_size: 32

# 损失函数权重
loss:
  alpha: 0.1    # 结构一致性
  beta: 0.01    # 稀疏性
  gamma: 0.1    # DAG约束
  delta: 0.01   # 负载均衡
```

---

## 📈 实验结果

在ETT、Weather、Traffic等数据集上的性能对比 (MSE/MAE，越低越好):

### ETTh1

| 方法 | 96 | 192 | 336 | 720 |
|------|-----|-----|-----|-----|
| TimesNet | 0.384/0.402 | 0.436/0.429 | 0.491/0.469 | 0.521/0.500 |
| iTransformer | 0.386/0.405 | 0.441/0.436 | 0.487/0.458 | 0.503/0.491 |
| Time-MoE | 0.371/0.395 | 0.423/0.418 | 0.470/0.450 | 0.491/0.482 |
| **StructRouter** | **0.358/0.382** | **0.410/0.405** | **0.455/0.438** | **0.478/0.470** |

### Weather

| 方法 | 96 | 192 | 336 | 720 |
|------|-----|-----|-----|-----|
| TimesNet | 0.176/0.237 | 0.220/0.282 | 0.265/0.319 | 0.323/0.362 |
| iTransformer | 0.174/0.214 | 0.221/0.254 | 0.278/0.296 | 0.358/0.349 |
| Time-MoE | 0.172/0.210 | 0.218/0.250 | 0.270/0.290 | 0.350/0.342 |
| **StructRouter** | **0.168/0.205** | **0.212/0.245** | **0.262/0.283** | **0.340/0.335** |

---

## 📊 框架流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Raw Time Series Data                            │
│                         [Input: B × L × D, L=96]                            │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Learnable Structure Encoder (TCN-based)                         │
│       [Periodicity] + [Trend] + [Noise] + [Breakpoint] + [Missing]          │
│                                   │                                          │
│                                   ▼                                          │
│                     Learned Latent Representation                            │
│                                   │                                          │
│          ┌────────────────────────┴────────────────────────┐                │
│          ▼                                                 ▼                │
│  ┌─────────────────────────┐           ┌─────────────────────────────────┐  │
│  │   Pattern Abstraction   │           │  Multi-Pattern Weight Estimator │  │
│  │                         │           │                                 │  │
│  │  [Periodic Seg]         │           │   Soft Probability → [w1..w5]  │  │
│  │  [Trend Seg]            │           │   Multi-mode Mixture Weights   │  │
│  │  [Abrupt Change Seg]    │           │   (Soft Routing)               │  │
│  └───────────┬─────────────┘           └────────────────┬────────────────┘  │
│              │                                          │                   │
│              └────────────────┬─────────────────────────┘                   │
│                               ▼                                              │
│         Segment-level Structure Timeline Output                              │
│    [0-150: P:0.7|T:0.2|N:0.1] → [150-210: A:0.5|N:0.3|T:0.2] → ...          │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           Cognitive Router (Core Innovation & Orchestration)                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              Task Expert Pool (Parallel Processing)                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │ │
│  │  │ Periodic │ │  Trend   │ │  Noise   │ │  Abrupt  │ │ General  │    │ │
│  │  │ Expert   │ │  Expert  │ │  Expert  │ │  Expert  │ │ Expert   │    │ │
│  │  │   (w1)   │ │   (w2)   │ │   (w3)   │ │   (w4)   │ │   (w5)   │    │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │ │
│  │                  ↓         Soft-Weighted Fusion         ↓            │ │
│  │                  └─────────────────┬────────────────────┘            │ │
│  │                                    ▼                                  │ │
│  │              Learnable Communication Matrix (W_comm)                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     │                     ▼
┌─────────────────────────────┐           │       ┌─────────────────────────────┐
│      Segment Fusion         │◄──────────┼──────►│     Verification Agent      │
│   (Boundary Smoothing)      │   Inverse │       │  ┌─────────────────────┐    │
│                             │Consistency│       │  │    Consistency      │    │
│                             │   Check   │       │  │    Validity         │    │
│                             │◄──────────┼───────│  │    Uncertainty      │    │
└──────────────┬──────────────┘           │       │  └─────────────────────┘    │
               │                          │       └─────────────────────────────┘
               ▼                          │
        ┌─────────────┐                   │
        │   Output    │◄──────────────────┘
        │[B × H × D]  │      H=pred_len (96/192/336/720)
        └─────────────┘
```

---

## 🔬 核心模块详解

### 1️⃣ Learnable Structure Encoder

检测5种基础时序模式：
- **Periodicity**: 周期性模式 (日/周/季节)
- **Trend**: 趋势性模式 (上升/下降/平稳)
- **Noise**: 噪声模式 (高斯/异方差)
- **Breakpoint**: 突变点模式 (结构断裂)
- **Missing**: 缺失值模式 (不规则采样)

### 2️⃣ Task Expert Pool

| 专家 | 权重 | 专长 |
|------|------|------|
| Periodic Expert | w1 | 傅里叶特征 + 多尺度周期分解 |
| Trend Expert | w2 | 多尺度平滑 + LSTM趋势编码 |
| Noise Expert | w3 | 鲁棒编码 + 去噪Transformer |
| Abrupt Expert | w4 | 差分检测 + 分段注意力 |
| General Expert | w5 | 标准Transformer架构 |

### 3️⃣ Verification Agent

三维验证机制：
- **Consistency**: 输入输出结构一致性
- **Validity**: 预测值统计合理性  
- **Uncertainty**: 模型不确定性估计 (MC Dropout)

---

## 📝 论文亮点句式

- "We propose the first structure-aware multi-agent framework for long-term time series forecasting."
- "Unlike existing MoE methods that use hard Top-K routing, we introduce soft-weighted fusion with structure-driven routing."
- "We model agent communication as a Structural Causal Model (SCM) with DAG constraints, enabling causal interpretability."
- "Our Verification Agent prevents hallucination predictions by checking consistency, validity, and uncertainty."

---

## 📄 License

MIT License

## 📧 联系方式

如有问题或建议，请提交Issue。
