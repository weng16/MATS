# StructRouter: Contributions

ICML/NeurIPS-style itemized contributions.

---

## 1. Multi-Pattern Weight Estimation with Prototype Similarity (C1 + C2)

We propose a multi-pattern weight estimator that fuses MLP-based routing with prototype similarity: \(w = \text{Softmax}(\text{MLP}(z) + \text{softplus}(\lambda) \cdot \text{Sim}(z, P))\), where \(P \in \mathbb{R}^{K \times H}\) are learnable, orthogonal pattern prototypes. Unlike MoE (Shazeer et al.) and Switch Transformer, which rely on purely data-driven routing, our design provides an inductive bias that similar structural patterns receive similar routing weights. Combined with soft-weighted expert fusion (\(\text{Output} = \sum_i w_i \cdot \text{Expert}_i(x)\)), all five experts contribute in every forward pass, avoiding expert collapse and preserving full information for mixed-mode time series.

---

## 2. SCM Causal Communication with DAG Constraint (C4)

We model expert (agent) communication as a Structural Causal Model with a learnable adjacency matrix \(W_{\text{comm}}\) constrained to a Directed Acyclic Graph via the NOTEARS formulation: \(h(W) = \text{tr}(\exp(W \odot W)) - N = 0\). This yields interpretable causal structure over experts—which expert influences which—and supports do-calculus and counterfactual reasoning. Unlike standard attention or fully connected communication, the DAG constraint ensures acyclicity and parsimony through sparse regularization, providing both performance and explainability.

---

## 3. Closed-Loop Verification with Gradient-Aware Rerouting (C6)

We introduce a verification agent that performs three-dimensional checks: Consistency (input–output structure alignment), Validity (statistical and range checks on predictions), and Uncertainty (aleatoric and epistemic via Monte Carlo dropout). When verification fails, an Inverse Consistency Check computes a suggested weight adjustment and triggers gradient-aware rerouting. The output structure retains gradients so that the consistency loss backpropagates through the prediction path, enabling the model to correct routing decisions in a closed loop rather than relying solely on open-loop supervised signals.

---

## 4. Comprehensive Empirical Evaluation

We conduct extensive experiments on seven benchmark datasets—ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, and Traffic—under standard forecasting protocols (prediction lengths 96, 192, 336, 720). [TODO: fill after experiments] We provide ablations on soft vs. hard routing, DAG constraint, verification, and three-stage training, demonstrating the contribution of each component. The learned pattern prototypes and causal graphs offer interpretable insights into model behavior across domains.
