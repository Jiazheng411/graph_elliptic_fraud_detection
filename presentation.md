---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 28px;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #3498db;
  }
  table {
    font-size: 20px;
  }
---

# Bitcoin Fraud Detection with the Elliptic Dataset
## CS5284 Project

**Team Members:** Li Jiayi, Russell Loh Chun Fa, Zhang Jiazheng

---

# Agenda

1. **Introduction and Motivation**
2. **Dataset Exploration and Analysis**
   - Dataset Overview
   - Key Data Observations
3. **Methodology and Experiments**
   - Train-Test Split
   - Traditional Machine Learning Models
   - Graph Neural Networks
   - Self-Supervised Learning Approaches
4. **Results and Discussion**
   - Performance Comparison
   - Model Analysis
   - Temporal Evaluation
5. **Conclusion and Future Work**

---

# Introduction & Motivation

**The Problem:**
- Bitcoin transactions form a **graph network**
- Criminals use complex, multi-step transactions for money laundering
- Traditional fraud detection struggles with network structure

**Our Goal:**
Build classification models to predict **licit** vs **illicit** transactions using:
- Node features (transaction characteristics)
- Graph structure (network topology)

**Dataset:** Elliptic dataset (MIT, IBM, Elliptic company)

---

# Dataset Overview

**Real Bitcoin transaction data over ~1 year (49 time steps)**

| Component | Details |
|-----------|---------|
| **Nodes** | 203,769 transactions |
| **Edges** | 234,355 directed payment flows |
| **Features** | 166 per transaction |
| **Labels** | '1' = Illicit, '2' = Licit, 'unknown' |

**Major Challenges:**
- Severe class imbalance
- 77% unlabeled data
- Temporal dynamics

---

# Key Observation 1: Class Imbalance

| Class | Count | Percentage |
|-------|-------|------------|
| Unknown | 157,205 | **77.1%** |
| Licit (2) | 42,019 | 20.6% |
| Illicit (1) | 4,545 | **2.3%** |

**Implications:**
- Extreme imbalance: ~4,500 illicit vs ~42,000 licit
- Must effectively utilize unlabeled data
- Need careful evaluation metrics (F1-score, not just accuracy)

---

# Key Observation 2: Temporal Dynamics

**Irregular fraud patterns across time steps:**
- Fraud occurs in organized bursts, not steady background activity
- **Critical event at time step 43:** Major dark market closure
- Fundamentally changed fraud patterns post-closure

![width:700px](images/temporal_fraud_patterns.png)

---

# Key Observation 3: Network Structure

**Scale-Free Network:**
- Most transactions have few connections
- Few massive "hub" nodes (exchanges, large wallets)
- Power-law degree distribution

![width:700px](images/degree_distribution.png)

---

# Key Observation 4: Structural Differences

**Licit vs Illicit Transaction Networks:**

1. **Fragmentation:** Licit = dense clusters, Illicit = long chains + fragments
2. **Money laundering chains:** Long linear paths to obfuscate source
3. **Local isolation:** Illicit nodes have fewer immediate neighbors

![width:700px](images/largest_components.png)

---

# Methodology: Temporal Split

**Why temporal split (not random)?**
1. **Avoid data leakage** - random split leaks future info
2. **Real-world deployment** - models must predict future fraud
3. **Follow original paper** (Weber et al. 2019)

| Split | Time Steps | Nodes | Labeled | Edges |
|-------|------------|-------|---------|-------|
| **Train** | 1-34 | 136,265 | 29,894 | 156,843 |
| **Test** | 35-49 | 67,504 | 16,670 | 77,512 |

**No validation set:** Limited data + time step 43 market event

---

# Approaches Overview

**Model Categories Explored:**

| Category | Models |
|----------|--------|
| **Traditional ML** | Random Forest, XGBoost |
| **Graph Neural Networks** | GCN, GraphSAGE, GAT, GATv2, Graph Transformer |
| **Self-Supervised Learning** | BGRL+GCN, GAE+GCN, Local GAE+GCN, CoLA+GCN |

**Training Scenarios:**
- **Labeled only:** Train on ~30K labeled nodes (clean training)
- **With unknown:** Train on full dataset (~106K nodes, semi-supervised)

---

# Approaches: Traditional ML

**Random Forest** (Baseline from paper)
- Ensemble of decision trees with majority voting for classification
- 50 trees, max 50 features per split
- Uses only node features (no graph structure)
- **Results:** Precision = 0.91, Recall = 0.73, **F1 = 0.81**

**XGBoost**
- Gradient boosting: sequentially trains trees to correct previous errors
- Optimizes: $\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$ (loss + regularization)
- **Results:** [Data not in provided results]

**Limitation:** Cannot leverage graph structure, but strong baseline performance

---

# GNN Approach 1: Graph Convolutional Network (GCN)

**Core Idea:** Aggregate neighbor features via graph convolution

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{d_i d_j}} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)}\right)$$

where $\mathcal{N}(i)$ = neighbors of node $i$, $d_i$ = degree, $\mathbf{W}$ = learnable weights

**Architecture:**
- 2-layer GraphConv, hidden dim = 100
- Standard GNN baseline

**Training:** Labeled-only vs All nodes (with unknown)

| Training Mode | Precision | Recall | **F1** | Before t43 | After t43 |
|---------------|-----------|--------|--------|------------|-----------|
| Labeled only | 0.78 | 0.54 | **0.63** | 0.72 | 0.00 |
| All nodes | 0.71 | 0.46 | **0.56** | 0.72 | 0.01 |

**Key finding:** Labeled-only training performs better than semi-supervised

---

# GNN Approach 2: GraphSAGE (Russell)

**Core Idea:** Sample fixed-size neighborhoods and aggregate with learnable functions

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W} \cdot \text{CONCAT}\left(\mathbf{h}_i^{(l)}, \text{AGG}\left(\{\mathbf{h}_j^{(l)}, j \in \mathcal{N}_{\text{sample}}(i)\}\right)\right)\right)$$

Samples $u$ neighbors (with replacement if needed); aggregators learn different patterns

**Architecture:**
- Sample size = 2, Multiple aggregators: Mean, MaxPool, LSTM
- 2 layers, 500 epochs

**Results:**

| Aggregator | Precision | Recall | **F1** |
|------------|-----------|--------|--------|
| **LSTM** | 0.77 | 0.50 | **0.61** |
| Mean | 0.55 | 0.56 | **0.55** |
| MaxPool | 0.55 | 0.56 | **0.55** |

**Key finding:** LSTM aggregator performs best, capturing sequential patterns

---

# GNN Approach 3: Graph Attention Networks (Jiazheng)

**Core Idea:** Learn attention weights to determine neighbor importance

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\cdot)}, \quad \mathbf{h}_i' = \sigma\left(\sum_{j} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)$$

Multi-head attention captures different relationship aspects simultaneously

---

# GAT Results: Labeled Only Training

**GAT Variants:**

| Model | Precision | Recall | **F1** | Before t43 | After t43 |
|-------|-----------|--------|--------|------------|-----------|
| GAT Base | 0.58 | 0.64 | **0.61** | 0.69 | 0.01 |
| GAT 2L (MLP) | 0.70 | 0.62 | **0.66** | 0.77 | 0.01 |
| GAT 2L + Residual | 0.86 | 0.56 | **0.68** | 0.77 | 0.01 |
| GAT 4L + Residual | 0.84 | 0.58 | **0.69** | 0.78 | 0.02 |

**GATv2 Variants:**

| Model | Precision | Recall | **F1** | Before t43 | After t43 |
|-------|-----------|--------|--------|------------|-----------|
| GATv2 Base | 0.54 | 0.62 | **0.57** | 0.65 | 0.00 |
| GATv2 2L (MLP) | 0.72 | 0.63 | **0.67** | 0.76 | 0.02 |
| GATv2 2L + Residual | 0.86 | 0.57 | **0.68** | 0.77 | 0.00 |
| GATv2 4L + Residual | 0.78 | 0.61 | **0.69** | 0.79 | 0.02 |

---

# GAT Results: Training with Unknown Nodes

**Semi-supervised learning (includes unlabeled nodes):**

| Model | Precision | Recall | **F1** | Before t43 | After t43 |
|-------|-----------|--------|--------|------------|-----------|
| GAT 2L + Residual | 0.68 | 0.52 | **0.59** | 0.67 | 0.02 |
| **GAT 4L + Residual** | 0.86 | 0.60 | **0.71** | 0.77 | 0.01 |
| GATv2 2L + Residual | 0.85 | 0.57 | **0.68** | 0.72 | 0.02 |

**Key findings:**
- 4-layer networks generally outperform 2-layer
- Residual connections + MLP classifier essential
- **Best GAT:** 4L + Residual with unknown (F1 = 0.71)

---

# GNN Approach 4: Graph Transformer (Jiazheng)

**Core Idea:** Apply transformer's scaled dot-product attention to graph structure

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q} = \mathbf{W}_Q\mathbf{h}_i$, $\mathbf{K} = \mathbf{W}_K\mathbf{h}_j$, $\mathbf{V} = \mathbf{W}_V\mathbf{h}_j$

Separate Q/K/V projections provide more expressiveness than GAT; captures longer-range dependencies

**Architecture:** Multi-head (8 heads × 16 dims), 2 layers with residuals

**Results:**

| Model | Precision | Recall | **F1** | Before t43 | After t43 |
|-------|-----------|--------|--------|------------|-----------|
| **Graph Transformer 2L** | 0.80 | 0.64 | **0.71** | 0.82 | 0.01 |

**Key finding:**
- Best overall GNN performance (F1 = 0.71)
- Strong before market closure (F1 = 0.82)
- Transformer architecture captures long-range dependencies effectively

---

# Approaches: Self-Supervised Learning

**Challenge:** Only 23% of data is labeled - can we leverage unlabeled nodes?

**Strategy:** Pre-train on unlabeled nodes → Fine-tune classifier on labeled nodes

---

# SSL Methods: Bootstrap & Contrastive Learning

| Method | Explanation | F1 | Before t43 |
|--------|-------------|-----|------------|
| **BGRL** | Bootstrap Your Own Latent: contrastive learning without negative samples | **0.58** | 0.75 |
| BGRL Node Score | Use BGRL scores as features | 0.58 | 0.75 |
| **CoLA** | Contrastive multi-view learning with feature corruption | 0.53 | 0.62 |

---

# SSL Methods: Graph Autoencoders

| Method | Explanation | F1 | Before t43 |
|--------|-------------|-----|------------|
| **GAE** | Reconstructs full graph adjacency matrix | 0.40 | 0.64 |
| **GAE Node Score** | Uses GAE reconstruction error as features | **0.59** | 0.77 |
| **Local GAE** | Reconstructs only local k-hop neighborhoods | 0.55 | 0.71 |

**Key finding:** SSL methods underperform supervised GNNs (best SSL F1 = 0.59 vs GNN F1 = 0.71)

**Why?** Labeled data quality > unlabeled quantity for this task

---

# Results: Overall Comparison

| Approach | Model | Overall F1 | Before t43 | After t43 |
|----------|-------|------------|------------|-----------|
| **Traditional ML** | Random Forest | **0.81** | **0.90** | 0.03 |
| **GNN** | GCN (Labeled) | 0.63 | 0.72 | 0.00 |
| | GCN (All nodes) | 0.56 | 0.72 | 0.01 |
| | GraphSAGE (LSTM) | 0.61 | — | — |
| | GAT 4L + Residual | 0.69 | 0.78 | 0.02 |
| | GAT 4L w/ Unknown | **0.71** | 0.77 | 0.01 |
| | **Graph Transformer** | **0.71** | **0.82** | 0.01 |
| **SSL** | BGRL + Raw | 0.58 | 0.75 | 0.01 |
| | GAE Node Score | 0.59 | 0.77 | 0.02 |

**Key insight:** Traditional ML (Random Forest) achieves highest F1, but GNNs offer better interpretability and leverage graph structure

---

# Key Findings: Temporal Performance Drop

**Time Step 43 Market Event Impact:**

| Model | F1 Before (t35-42) | F1 After (t43-49) | **Drop** |
|-------|-------------------|-------------------|----------|
| Random Forest | 0.90 | 0.03 | **-97%** |
| GCN Labeled | 0.72 | 0.00 | **-100%** |
| Graph Transformer | 0.82 | 0.01 | **-99%** |
| GAT 4L w/ Unknown | 0.77 | 0.01 | **-99%** |

**Observations:**
- **Catastrophic performance drop** at time step 43 across ALL models
- Dark market closure fundamentally changed fraud patterns
- Models trained on pre-closure data cannot generalize
- Highlights critical need for continual learning / temporal adaptation

---

# Conclusion & Key Takeaways

**Main Contributions:**
1. ✅ Rigorous temporal split (no data leakage)
2. ✅ Comprehensive model comparison (Traditional ML, GNN, SSL)
3. ✅ Deep architectural exploration (12+ model variants)
4. ✅ Temporal analysis revealing market event impact

**Best Performing Models:**
- **Traditional ML:** Random Forest (F1 = 0.81, Before closure = 0.90)
- **GNN:** Graph Transformer (F1 = 0.71, Before closure = 0.82)
- **Best with Unknown nodes:** GAT 4L (F1 = 0.71, leverages unlabeled data)

**Major Challenge:**
- All models fail catastrophically after market regime change (t=43)
- Pre-event F1 ~0.70-0.90 → Post-event F1 ~0.00-0.03

---

# Future Work

1. **Temporal Adaptation:**
   - Continual learning approaches
   - Online model updates
   - Transfer learning across market regimes

2. **Advanced Architectures:**
   - Larger-scale Graph Transformers
   - Temporal GNNs (recurrent architectures)
   - Explainable AI for fraud detection

3. **Data Utilization:**
   - Better SSL methods for unlabeled data
   - Active learning for selective labeling
   - Multi-task learning with auxiliary tasks

---

# Questions?

**Thank you!**

**Team:** Li Jiayi, Russell Loh Chun Fa, Zhang Jiazheng

**Project:** Bitcoin Fraud Detection with Graph Neural Networks

---

# Backup: Technical Details

**Training Hyperparameters (GAT):**
- Hidden dim: 128, Heads: 8
- Dropout: 0.1 (feature, attention, residual)
- Optimizer: Adam (lr=1e-3, weight decay=5e-4)
- Loss: Cross-entropy with class weights [0.7, 0.3]
- Epochs: 1,000 (eval every 100)

**GraphSAGE Configuration:**
- 2 layers, Hidden dim: 128
- Aggregators: Mean, MaxPool, LSTM
- Neighbor sampling: u ∈ {1,2,3}
- Training epochs: 500

