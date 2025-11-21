---
marp: true
theme: default
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .small { font-size: 0.8em; }
  .center { text-align: center; }
  .highlight { background-color: #ffeb3b; padding: 2px 4px; }
size: 16:9
paginate: true
---

# Bitcoin Fraud Detection with the Elliptic Dataset
## CS5284 Project Report

**Project Group 07**
- Li Jiayi
- Russell Loh Chun Fa  
- Zhang Jiazheng

---

# Introduction and Motivation

- **The Problem:** Money laundering involves complex, multi-step transaction chains to obfuscate the source of illicit funds.

- **Context:** Bitcoin's public ledger offers transparency, yet pseudonymity makes fraud detection difficult.

- **Objective:** Classify transactions as **Licit** or **Illicit** using:
  - **Local Features:** Transaction fees, time steps, input/output values
  - **Network Topology:** Graph structure connecting transactions via fund flows

- **Core Challenge:** Can Graph Neural Networks (GNNs) and Self-Supervised Learning (SSL) outperform traditional classifiers on highly imbalanced, temporal financial data?

---

# Dataset Exploration: The Elliptic Data

<div class="columns">
<div>

**Graph Statistics:**
- Nodes: 203,769 (Transactions)
- Edges: 234,355 (Payment flows) 
- Temporal span: 49 time steps

**Key Observations:**
1. **Imbalance:** Only 2% are Illicit; 77% are Unknown
2. **Scale-Free:** Few hubs with massive connectivity
3. **Illicit Topology:** Fraud often forms long linear chains or small fragmented clusters to evade tracking

</div>
<div>

![w:400](images/largest_components.png)
*Largest components of illicit and licit channels*

![w:400](images/temporal_fraud_patterns.png)
*Temporal distribution of Illicit transactions*

</div>
</div>

---

# Methodology: Experimental Setup

## Data Split Strategy
We utilized a **Temporal Split** to mimic real-world deployment and prevent data leakage:
- **Train:** Time steps 1–34 (136k nodes)
- **Test:** Time steps 35–49 (67k nodes)
- No random shuffling; future data remains unseen

## Traditional Baselines
- **Random Forest (RF):** Leverages 166 handcrafted features
- **Performance:** The RF sets a very high bar with an Illicit F1-score of ≈ 0.81
- **Insight:** The raw tabular features provided by Elliptic are highly discriminative even without explicit graph propagation

---

# Graph Neural Networks: GCN & GraphSAGE

To capture neighborhood dependencies, we deployed standard GNN architectures:

**Graph Convolutional Networks (GCN):**
- Tested two regimes: Training on *Labeled Only* vs. *All Nodes* (using masks for unknowns)
- **Result:** Performance degrades when including unknown nodes, suggesting label noise or irrelevant structural signals in the unlabeled majority

**GraphSAGE (Sample and Aggregate):**
- Evaluated Mean, MaxPool, and LSTM aggregators
- **Best Variant:** LSTM aggregator yielded the highest F1 among SAGE models (0.61)
- **Limitation:** Fixed aggregation often under-utilizes the heterogeneity of transaction neighborhoods compared to attention mechanisms

---

# Advanced Architectures: Attention Mechanisms

We explored models that dynamically weight neighbor importance:

**Graph Attention Networks (GAT):**
- Uses multi-head attention to focus on high-risk neighbors
- A 4-layer GATv2 with residual connections achieved an F1 of **0.794** (pre-market closure)

**Graph Transformer:**
- Adapts the Transformer architecture to graphs with Q/K/V projections and global attention capabilities
- **Result:** Achieved the **best overall performance among deep models** (F1 = 0.817 pre-closure)
- Effectively models long-range dependencies typical of money laundering chains

---

# Self-Supervised Learning Approaches

Given the 77% unlabeled data, we hypothesized SSL could extract robust representations.

1. **BGRL (Bootstrapped Graph Latents):** Uses interacting online/target encoders with augmentation
2. **Graph Autoencoder (GAE):** Reconstruction-based anomaly detection (Global and Local subgraph variants)  
3. **CoLA (Contrastive Learning for Anomalies):** Maximizes agreement between target nodes and their local subgraphs

**Findings:**
- SSL methods generally improved *recall* but sacrificed precision
- None outperformed the purely supervised GCN baseline
- **Conclusion:** Generic reconstruction tasks may smooth embeddings too much, blurring the sharp decision boundaries needed for rare illicit nodes

---

# Results: Performance Comparison (Illicit Class)

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Random Forest** | **0.91** | 0.72 | **0.81** |
| Graph Transformer | 0.80 | 0.64 | 0.71 |
| GATv2 (4-layer) | 0.86 | 0.60 | 0.69 |
| GCN (Labeled Only) | 0.78 | 0.55 | 0.63 |
| GraphSAGE (LSTM) | 0.77 | 0.51 | 0.61 |
| SSL (BGRL+GCN) | 0.70 | 0.50 | 0.58 |

<div class="small">

*Metrics evaluated on Test Set (weighted towards pre-closure data)*

</div>

**Analysis:**
- Tree-based models dominate on tabular features
- Among GNNs, **Attention is critical**. Graph Transformer approaches RF performance  
- Deep models struggle to balance high precision with recall due to extreme class imbalance

---

# Temporal Analysis: The Dark Market Shutdown

<div class="columns">
<div>

A major event occurred at **Timestep 43** (Dark Market Shutdown).

- **Observation:** All models perform well for t < 43
- **Collapse:** At t ≥ 43, F1 scores drop to near zero across the board
- **Reason:** 
  1. **Concept Drift:** Fraud patterns changed fundamentally after the shutdown
  2. **Sparsity:** The volume of labeled illicit nodes drops significantly in the final steps

</div>
<div>

![w:400](images/models_f1_comparison.png)
*Model average F1 scores*

![w:400](images/time_eval.png)
*Model F1 scores over time*

</div>
</div>

---

# Conclusion and Future Work

**Summary:**
- We successfully implemented a suite of GNNs and SSL pipelines for Bitcoin fraud detection
- **Graph Transformer** proved to be the most capable neural architecture, leveraging global attention
- Traditional Random Forests remain a formidable baseline due to high-quality feature engineering

**Future Directions:**
- **Temporal GNNs:** Use TGNs or TGATs to explicitly model time and handle concept drift (the T=43 crash)
- **Hybrid Models:** Combine RF feature selection with Graph Transformer structural learning
- **Active Learning:** To address the label sparsity in post-shutdown epochs

---

# Thank You!
## Questions?

<div class="center">

**Project Group 07**
CS5284 - Graph Machine Learning
National University of Singapore

</div>