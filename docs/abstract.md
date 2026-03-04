# StructRouter: Structure-Aware Multi-Agent Time Series Forecasting

**Target Venue:** ACM MM 2026

---

## Abstract

Time series forecasting remains challenging when data exhibit mixed structural patterns—periodicity, trends, noise, abrupt changes—that vary across segments. Existing Mixture-of-Experts and Transformer approaches use hard routing that activates only a subset of experts, discarding information and risking expert collapse. We propose **StructRouter**, a structure-aware multi-agent framework that models time series as a mixture of learnable structural patterns with soft routing and interpretable causal communication.

StructRouter is the first to combine learnable pattern prototypes with soft routing: \(w = \text{Softmax}(\text{MLP}(z) + \text{softplus}(\lambda) \cdot \text{Sim}(z, P))\), where \(P\) are orthogonal prototypes. All five experts (Periodic, Trend, Noise, Abrupt, General) contribute in every forward pass: \(\text{Output} = \sum_i w_i \cdot \text{Expert}_i(x)\). Agent communication is modeled as an SCM with DAG constraint \(h(W) = \text{tr}(\exp(W \odot W)) - N = 0\), yielding interpretable causal structure. A verification agent performs three-dimensional checks (Consistency, Validity, Uncertainty) and gradient-aware rerouting when predictions violate structure constraints. Training proceeds in three stages: self-supervised pretraining, end-to-end joint loss fine-tuning, and PPO-based reinforcement fine-tuning for routing optimization.

We evaluate on ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, and Traffic. [TODO: fill after experiments] StructRouter achieves [TODO: fill after experiments] with improved interpretability through learned prototypes and causal graphs.

**Keywords:** Time series forecasting, mixture of experts, soft routing, structural causal model, DAG constraint, reinforcement fine-tuning, verification agent.
