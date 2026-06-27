# VigilNet — AML Detection System

> **Real-time Anti-Money-Laundering detection using Graph Neural Networks, Transformer-based behaviour encoding, and adaptive multi-modal fusion.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [1 — Offline: Graph Construction](#1--offline-graph-construction)
  - [2 — Behaviour Encoder Training](#2--behaviour-encoder-training)
  - [3 — GATv2 Graph Embedding Training](#3--gatv2-graph-embedding-training)
  - [4 — Fusion Network Training](#4--fusion-network-training)
  - [5 — Online Inference Pipeline](#5--online-inference-pipeline)
- [Models](#models)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Overview

VigilNet is a graph-aware AML detection system built for large-scale financial transaction networks. It combines three complementary signals to flag suspicious accounts:

| Signal | Source | Model |
|--------|--------|-------|
| **Local behaviour** | Per-account transaction sequences | Transformer Encoder |
| **Graph topology** | Account–transaction network | GATv2 (Graph Attention v2) |
| **Risk memory** | Persistent community / PageRank scores | MLP Projection |

These signals are fused by an **Adaptive Gated Fusion Network** that learns per-account weighting between local and graph representations before producing a fraud probability.

---

## Architecture

```
Dataset (HI-Small IBM AML)
        │
        ▼
┌──────────────────────────────────────────────────┐
│               OFFLINE PIPELINE                   │
│                                                  │
│  Graph Builder  ──►  AML Edge Weight Engine      │
│       │                       │                  │
│       ▼                       ▼                  │
│  Risk Diffusion (PageRank)    │                  │
│       │                       │                  │
│       ▼                       │                  │
│  Community Intelligence ◄─────┘                  │
│       │                                          │
│       ▼                                          │
│  Persistent Risk Memory                          │
└──────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
┌─────────────────┐                 ┌─────────────────────┐
│ Behaviour Stage │                 │   GNN Stage         │
│                 │                 │                     │
│ Transaction     │                 │  GATv2Conv (L1)     │
│ Sequences       │                 │  + Residual + BN    │
│       │         │                 │        │            │
│ Transformer     │                 │  GATv2Conv (L2)     │
│ Encoder         │                 │  + Residual + BN    │
│       │         │                 │        │            │
│ Behaviour       │                 │  Graph Embeddings   │
│ Embeddings      │                 │  (128-d, L2-norm)   │
└────────┬────────┘                 └──────────┬──────────┘
         │                                     │
         ▼                                     ▼
┌──────────────────────────────────────────────────┐
│              FUSION NETWORK                      │
│                                                  │
│  Memory Projection  +  Behaviour  ──►  Local     │
│                                                  │
│  Adaptive Gate  =  σ( W · [local ‖ graph] )      │
│                                                  │
│  fused = gate·graph  +  (1−gate)·local           │
│       │                                          │
│  MLP Classifier  ──►  Fraud Probability          │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
AML-System/
│
├── behaviour/                   # Transaction sequence modelling
│   ├── behaviour_dataset.py     # PyTorch Dataset for sequences
│   ├── transaction_sequence_builder.py
│   ├── positional_encoding.py
│   ├── transformer_encoder.py   # BehaviourEncoder model
│   ├── extract_embeddings.py    # Saves behaviour_embeddings.pkl
│   └── train_transformer.py     # Training script
│
├── graph/                       # Graph construction & analytics
│   ├── build_graph.py           # Builds NetworkX graph from CSV
│   ├── edge_weight.py           # AML-aware edge weight engine
│   ├── aml_pagerank.py          # Risk diffusion (personalised PageRank)
│   ├── community_engine.py      # Louvain community features
│   ├── node_risk_initializer.py # Seed risk scores
│   ├── persistent_risk_memory.py
│   ├── validate_graph.py
│   └── config.py
│
├── gnn/                         # Graph Neural Network stage
│   ├── gatv2_model.py           # GATv2Model (2-layer with residuals)
│   └── train_gat.py             # Mini-batch training w/ ManualNeighborSampler
│
├── representation/              # Shared representation modules
│   ├── memory_projection.py     # Projects 6-d risk memory → 128-d
│   └── feature_fusion.py        # Concatenates memory + behaviour → 128-d
│
├── fusion/                      # Multi-modal fusion stage
│   ├── fusion_network.py        # FusionNetwork (adaptive gated MLP)
│   └── train_fusion.py          # Fusion training script
│
├── data/
│   ├── raw/                     # Source CSV / pattern files  [git-ignored]
│   └── processed/               # Intermediate .pkl / .pt artifacts [git-ignored]
│
├── models/                      # Saved model weights  [git-ignored]
│   ├── behaviour_encoder.pth
│   ├── gatv2_model.pth
│   └── fusion_network.pth
│
├── docs/                        # Design documents
│   ├── ARCHITECTURE.md
│   ├── DATA_DICTIONARY.md
│   └── GRAPH_SPECIFICATIONS.md
│
├── online/                      # Online inference pipeline
│   ├── online_inference.py     # OnlineAMLSystem — dual-account scoring
│   ├── transaction_processor.py # Maintains per-account transaction sequences
│   ├── graph_updater.py        # Adds edges & recomputes edge weights live
│   └── risk_memory_updater.py  # PageRank + community risk memory refresh
│
├── api/                         # Inference API (WIP)
├── offline/                     # Offline pipeline orchestration (WIP)
├── notebooks/                   # Exploratory notebooks
├── utils/                       # Shared utilities
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline

### 1 - Offline: Graph Construction

Builds the account-level transaction graph and computes AML-specific analytics.

```bash
# Build base graph from raw CSV
python graph/build_graph.py

# Compute AML edge weights
python graph/edge_weight.py

# Risk diffusion (PageRank)
python graph/aml_pagerank.py

# Community detection & features
python graph/community_engine.py

# Persistent risk memory
python graph/persistent_risk_memory.py
```

**Outputs** (saved to `data/processed/`):
- `graph.pkl` — raw NetworkX graph
- `weighted_graph.pkl` — graph with AML edge weights
- `pagerank_scores.pkl` — per-node risk scores
- `community_features.pkl` — Louvain community features
- `risk_memory.pkl` — merged risk memory per account

---

### 2 - Behaviour Encoder Training

Trains a Transformer over per-account transaction sequences.

```bash
cd behaviour
python train_transformer.py
```

After training, extract embeddings for all accounts:

```bash
python extract_embeddings.py
```

**Output:** `data/processed/behaviour_embeddings.pkl`

---

### 3 - GATv2 Graph Embedding Training

Trains the 2-layer GATv2 model on the full account graph using mini-batch neighbourhood sampling.

```bash
python gnn/train_gat.py
```

- Uses `ManualNeighborSampler` (no `pyg-lib` / `torch-sparse` required)
- Mixed-precision (AMP/fp16) when CUDA is available
- Saves embeddings to `data/processed/graph_embeddings.pt`

---

### 4 - Fusion Network Training

Trains the Adaptive Gated Fusion Network on pre-computed local + graph embeddings.

```bash
python fusion/train_fusion.py
```

---

### 5 — Online Inference Pipeline

Real-time transaction scoring. The system ingests a new transaction and evaluates **both** the sender and receiver to produce a per-transaction fraud probability.

#### Modules

| Module | File | Responsibility |
|--------|------|----------------|
| `OnlineAMLSystem` | `online/online_inference.py` | Orchestrates the full pipeline; exposes `predict()` |
| `TransactionProcessor` | `online/transaction_processor.py` | Maintains per-account transaction sequences (bounded deque) |
| `GraphUpdater` | `online/graph_updater.py` | Adds new edges to the live graph & recomputes AML edge weights |
| `RiskMemoryUpdater` | `online/risk_memory_updater.py` | Refreshes PageRank + community risk memory for queried accounts |

#### Prediction Flow

```
New Transaction (sender, receiver, features)
        │
        ├──► TransactionProcessor.add_transaction()   # update sequences
        ├──► GraphUpdater.add_transaction()            # update live graph
        │
        ├──► predict_account(sender)
        │       ├── RiskMemoryUpdater.update(sender)
        │       ├── BehaviourEncoder  →  behaviour embedding
        │       ├── MemoryProjection  →  memory embedding
        │       ├── FeatureFusion     →  local embedding
        │       ├── Graph Embedding   (precomputed lookup)
        │       └── FusionNetwork     →  sender_probability
        │
        ├──► predict_account(receiver)   # same pipeline
        │       └── → receiver_probability
        │
        └──► transaction_probability = max(sender, receiver)
             prediction = 1 if transaction_probability ≥ 0.93
```

#### Quick Start

```python
from online.online_inference import OnlineAMLSystem

aml = OnlineAMLSystem()

result = aml.predict(
    sender="121_8123FB9B0",
    receiver="10_8000EBD30",
    transaction={
        "amount": 38769.39,
        "timestamp": 202209010021,
        "payment_format": "Cheque",
        "currency": "US Dollar",
        "time_gap": 3600,
        "in_degree": 14,
        "out_degree": 9,
    }
)

print(result)
# {
#   "sender_probability": 0.12,
#   "receiver_probability": 0.87,
#   "transaction_probability": 0.87,
#   "prediction": 0
# }
```

#### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `sender_probability` | `float` | Fraud risk score of the sender account (0–1) |
| `receiver_probability` | `float` | Fraud risk score of the receiver account (0–1) |
| `transaction_probability` | `float` | `max(sender, receiver)` — overall transaction risk |
| `prediction` | `int` | `1` if `transaction_probability ≥ 0.93`, else `0` |

---

## Models

| Model | File | Architecture | Output |
|-------|------|-------------|--------|
| `BehaviourEncoder` | `behaviour/transformer_encoder.py` | Transformer Encoder | 128-d embedding |
| `GATv2Model` | `gnn/gatv2_model.py` | 2× GATv2Conv + residuals + BN | 128-d L2-norm embedding |
| `FusionNetwork` | `fusion/fusion_network.py` | Adaptive gate + MLP classifier | Fraud logit |

---

## Dataset

This project uses the **IBM AML (HI-Small) dataset**:

- `data/raw/HI-Small_Trans.csv` — raw transaction records
- `data/raw/HI-Small_Patterns.txt` — labelled laundering patterns

> **Note:** Raw data files are excluded from version control (`.gitignore`). Download from the [IBM AML dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and place in `data/raw/`.

**Graph Statistics:**
- **Nodes:** ~515,000 accounts
- **Edges:** ~1,015,000 transactions
- **Positive (laundering) accounts:** highly imbalanced (~5%)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/VigilNet.git
cd VigilNet/AML-System

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (match your CUDA version)
pip install torch-geometric
```

> See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for CUDA-specific wheels.

---

## Usage

Run the full pipeline end-to-end:

```bash
# 1. Build graph artifacts
python graph/build_graph.py
python graph/edge_weight.py
python graph/aml_pagerank.py
python graph/community_engine.py
python graph/persistent_risk_memory.py

# 2. Train behaviour encoder & extract embeddings
cd behaviour && python train_transformer.py && python extract_embeddings.py && cd ..

# 3. Train GATv2 & save graph embeddings
python gnn/train_gat.py

# 4. Train fusion network
python fusion/train_fusion.py

# 5. Run online inference (smoke test)
python online/online_inference.py
```

---

## Results

Evaluation performed on the **held-out test split** (77,264 accounts — 15% of the HI-Small dataset, never seen during training) using the trained Fusion Network at a decision threshold of **0.93**.

### Classification Metrics

| Metric    | Value   |
| --------- | ------: |
| Accuracy  | 97.55 % |
| Precision | 20.93 % |
| Recall    | 35.53 % |
| F1 Score  | 26.34 % |

### Ranking Metrics

| Metric  | Value  |
| ------- | -----: |
| ROC-AUC | 0.9208 |
| PR-AUC  | 0.1266 |

### Error Rates

| Metric                      | Value   |
| --------------------------- | ------: |
| False Positive Rate (FPR)   |  1.68 % |
| False Negative Rate (FNR)   | 64.47 % |

### Confusion Matrix

|                        | Predicted Negative | Predicted Positive |
| ---------------------- | -----------------: | -----------------: |
| **Actual Negative**    |        75,029 (TN) |         1,281 (FP) |
| **Actual Positive**    |           615 (FN) |           339 (TP) |

> **Note:** Metrics are computed on a stratified 15% test split that was held out during all training stages (Behaviour Encoder, GATv2, Fusion Network). The ROC-AUC of 0.921 on unseen accounts confirms genuine ranking ability. The elevated FNR reflects the conservative threshold (0.93) chosen to minimise false alerts in a highly imbalanced dataset (~1.2% positive rate).

---

## License

This project is for research and educational purposes.
Dataset © IBM — see original [dataset license](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
