# StructRouter: Concept Definitions

Formal definitions of the five main innovations (C1–C5) and supporting components (C6–C8) in the StructRouter framework for structure-aware multi-agent time series analysis.

---

## C1 — Multi-Pattern Weight Estimator

### [Formal Academic Definition]

The Multi-Pattern Weight Estimator is a routing module that assigns a probability distribution over \(K\) structural patterns (Periodic, Trend, Noise, Abrupt, General) to each input time series segment. Given a structure feature vector \(\mathbf{z} \in \mathbb{R}^H\) from a structure encoder, the estimator combines two complementary mechanisms: (1) an MLP that directly maps \(\mathbf{z}\) to routing logits, and (2) a prototype-based similarity term where \(\mathbf{P} \in \mathbb{R}^{K \times H}\) is a set of learnable, orthogonal pattern prototypes. The similarity \(\text{Sim}(\mathbf{z}, \mathbf{P})\) is computed as the cosine similarity between \(\mathbf{z}\) and each prototype (with an optional learned linear transform). A learnable scalar \(\lambda\) (passed through softplus for positivity) balances the two terms. The final weights are obtained via temperature-scaled softmax over the combined logits.

### [Distinction from Prior Work]

Unlike MoE (Shazeer et al.) and Switch Transformer, which use a single router network to produce discrete or sparse expert selection, StructRouter explicitly models time series as a *mixture of structural patterns* with learnable prototypes. The prototype similarity term provides an inductive bias: inputs with similar structure should receive similar routing weights. This differs from purely data-driven routing in that the prototypes act as interpretable anchors for pattern types (periodicity, trend, noise, etc.), and orthogonality regularization keeps them distinct.

### [Key Formula]

\[
\mathbf{w} = \text{Softmax}\left( \frac{\text{MLP}(\mathbf{z}) + \text{softplus}(\lambda) \cdot \text{Sim}(\mathbf{z}, \mathbf{P})}{\tau} \right), \quad \text{Sim}(\mathbf{z}, \mathbf{P}) = \mathbf{z}_{\text{norm}} \mathbf{P}_{\text{norm}}^\top
\]

where \(\mathbf{P}_{\text{norm}}\) and \(\mathbf{z}_{\text{norm}}\) are L2-normalized, \(\tau\) is the temperature, and \(\sum_k w_k = 1\).

---

## C2 — Soft-Weighted Expert Fusion

### [Formal Academic Definition]

Soft-Weighted Expert Fusion computes the model output as a convex combination of \(K=5\) specialized expert outputs, where each expert is designed for a distinct structural pattern (Periodic, Trend, Noise, Abrupt, General). Given input \(\mathbf{x} \in \mathbb{R}^{B \times L \times D}\) and routing weights \(\mathbf{w} \in \mathbb{R}^{B \times K}\), the fused output is the weighted sum of all expert outputs. All experts participate in every forward pass; gradients flow to all experts, promoting stable training and avoiding expert collapse. Optional components include expert-level dropout, load-balance regularization, and residual connections from the input.

### [Distinction from Prior Work]

Switch Transformer and related work use *hard* or *sparse* routing: only the top-1 or top-\(k\) experts are activated per token. StructRouter uses *soft* routing: all experts contribute proportionally to their weights. This preserves full information from all pattern types, reduces gradient variance, and avoids the routing instability and expert underutilization common in hard routing. The design is motivated by the observation that real-world time series often exhibit mixed patterns (e.g., periodic + trend), which soft fusion can represent more faithfully.

### [Key Formula]

\[
\text{Output} = \sum_{i=1}^{K} w_i \cdot \text{Expert}_i(\mathbf{x}) + \text{Residual}(\mathbf{x})
\]

with \(\sum_i w_i = 1\) and optional expert dropout and load-balance loss \(\mathcal{L}_{\text{balance}}\).

---

## C3 — Tool Adapter with RFT

### [Formal Academic Definition]

The Tool Adapter treats the weight estimator as a tunable "tool" and applies reinforcement fine-tuning (RFT) to optimize routing decisions for downstream task performance. A policy network takes the base routing weights (from the weight estimator) and structure features as input, and outputs a residual adjustment to the weights. The adjusted weights are obtained by adding the scaled residual to the base weights and applying softmax. A value head estimates state value for advantage computation. Training uses PPO with clipped surrogate objective, value loss, and entropy bonus. The reward combines task performance (e.g., negative prediction error) and structure consistency between input and output.

### [Distinction from Prior Work]

Standard MoE and Switch Transformer optimize routing only via supervised gradients. StructRouter adds a third training stage (RFT) where routing is explicitly optimized with respect to a reward signal. This allows the model to correct routing mistakes that supervised loss alone may not penalize directly. The residual formulation ensures that RFT refines rather than overwrites the pretrained routing policy.

### [Key Formula]

\[
\mathbf{w}_{\text{adj}} = \text{Softmax}\left( \mathbf{w}_{\text{base}} + \eta \cdot \pi_\theta(\mathbf{w}_{\text{base}}, \mathbf{z}) \right), \quad \mathcal{R} = \mathcal{R}_{\text{task}} + \alpha \mathcal{R}_{\text{consist}}
\]

where \(\pi_\theta\) is the policy network, \(\eta\) is a learnable residual scale, and PPO optimizes \(\mathbb{E}[\mathcal{R}]\) with GAE.

---

## C4 — SCM Causal Communication

### [Formal Academic Definition]

Agent (expert) communication is modeled as a Structural Causal Model (SCM). Each agent’s updated state is a weighted sum of other agents’ outputs, where the weights form a causal adjacency matrix \(\mathbf{W}_{\text{comm}} \in \mathbb{R}^{N \times N}\). The matrix is constrained to represent a Directed Acyclic Graph (DAG) via the NOTEARS formulation: \(h(\mathbf{W}) = \text{tr}(\exp(\mathbf{W} \odot \mathbf{W})) - N = 0\), which holds if and only if the graph is acyclic. In practice, \(\mathbf{W}\) is clamped to \([-2, 2]\) for numerical stability of the matrix exponential. Sparse regularization encourages a parsimonious causal graph.

### [Distinction from Prior Work]

Standard multi-agent or multi-expert architectures use attention or fully connected layers for communication, which do not enforce acyclicity or interpretable causal structure. StructRouter’s SCM formulation yields a DAG over agents, enabling causal interpretation (e.g., which expert influences which) and supporting do-calculus and counterfactual reasoning. The NOTEARS constraint is differentiable and avoids discrete graph search.

### [Key Formula]

\[
\text{Agent}_j^{\text{new}} = \text{Agent}_j + \text{gate} \cdot \text{Receive}\left( \sum_i W_{\text{comm}}[i,j] \cdot \text{Message}(\text{Agent}_i) \right), \quad h(\mathbf{W}) = \text{tr}\left(e^{\mathbf{W} \odot \mathbf{W}}\right) - N = 0
\]

with \(\mathbf{W}\) clamped to \([-2, 2]\) and diagonal masked for no self-loops.

---

## C5 — Temporal Causal Extension

### [Formal Academic Definition]

The temporal causal extension generalizes the SCM to the time dimension by introducing lag-based causal matrices. The state of agent \(j\) at time \(t+1\) depends on the states of all agents at the current and past time steps. A causal tensor \(\mathbf{W}_{\text{temporal}} \in \mathbb{R}^{N \times N \times \tau_{\max}}\) encodes the strength of influence from agent \(i\) to agent \(j\) at lag \(\tau\). The instantaneous component \(\mathbf{W}_{\text{instant}}\) is subject to the same DAG constraint as in C4; lagged components naturally respect temporal causality (future cannot influence past). A learnable decay factor downweights older lags.

### [Distinction from Prior Work]

Prior causal or communication modules in time series models typically operate at a single time scale. The temporal extension explicitly models *when* one agent’s output affects another, supporting both instantaneous and lagged causal effects. This is aligned with Granger causality and temporal SCMs in econometrics.

### [Key Formula]

\[
\mathbf{X}_j(t+1) = \mathbf{X}_j(t) + \text{Proj}\left( \sum_i W_{\text{inst}}[i,j] \cdot \mathbf{X}_i(t) + \sum_{\tau=1}^{\tau_{\max}} \sum_i \gamma^\tau W_{\text{temp}}[i,j,\tau] \cdot \mathbf{X}_i(t-\tau) \right)
\]

where \(\gamma\) is the lag decay and \(h(\mathbf{W}_{\text{inst}}) = 0\) enforces DAG for the instantaneous graph.

---

## Supporting Components

### C6 — Verification Agent

A verification module performs three checks: (1) **Consistency**: cosine similarity and learned scoring between input and output structure weights; (2) **Validity**: statistical and range checks on predictions (mean, std, skewness, kurtosis, outlier ratio); (3) **Uncertainty**: aleatoric and epistemic uncertainty via Monte Carlo dropout. The **Inverse Consistency Check** computes a suggested weight adjustment when output structure deviates from input structure; this supports gradient-aware rerouting where consistency loss backpropagates through the prediction path. Rerouting applies the suggested weights only when verification fails.

### C7 — Joint Loss

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{consist}} + \beta \mathcal{L}_{\text{sparse}} + \gamma(t) \mathcal{L}_{\text{causal}} + \delta \mathcal{L}_{\text{balance}} + \varepsilon \mathcal{L}_{\text{ortho}}
\]

where \(\mathcal{L}_{\text{task}}\) is the downstream loss (e.g., MSE), \(\mathcal{L}_{\text{consist}}\) aligns input/output structure, \(\mathcal{L}_{\text{sparse}}\) encourages concentrated routing, \(\mathcal{L}_{\text{causal}} = h(\mathbf{W})^2\) enforces DAG, \(\mathcal{L}_{\text{balance}}\) prevents expert collapse, and \(\mathcal{L}_{\text{ortho}} = \|\mathbf{P}_{\text{norm}}^\top \mathbf{P}_{\text{norm}} - \mathbf{I}\|_F^2\) maintains prototype orthogonality. \(\gamma(t)\) uses DAG warmup (linear ramp) to avoid early training instability.

### C8 — Three-Stage Training

1. **Stage 1 (Self-Supervised)**: Contrastive learning (InfoNCE on augmented pairs), masked reconstruction, and structure prediction with pseudo-labels. Trains structure encoder, weight estimator, and reconstruction head.
2. **Stage 2 (End-to-End)**: Full model fine-tuning with JointLoss. Per-module learning rates: prototypes slower, \(\mathbf{W}_{\text{comm}}\) faster. DAG warmup and gradient clipping.
3. **Stage 3 (RFT)**: PPO-based reinforcement fine-tuning of the Tool Adapter and weight estimator. Reward = task performance + structure consistency.

### Convergence Aids

- **RevIN**: Reversible instance normalization (Kim et al., ICLR 2022) before encoder and after prediction head to handle distribution shift.
- **Per-module LRs**: Prototype parameters use \(0.1 \times\) base LR; \(\mathbf{W}_{\text{comm}}\) uses \(3 \times\) base LR.
- **Prototype orthogonality**: \(\mathcal{L}_{\text{ortho}}\) regularizes the Gram matrix of normalized prototypes toward identity.
