"""
evaluation/evaluate.py
────────────────────────────────────────────────────────────────────────────────
Full evaluation of the trained FusionNetwork on the complete dataset.

Configuration
─────────────
  THRESHOLD  — operating threshold for binary classification.
               Set to the value that optimises your chosen objective
               (e.g. 0.93 for best F1, 0.05 for best Recall).

Metrics reported
────────────────
  Accuracy · Precision · Recall · F1
  ROC-AUC  · PR-AUC
  False Positive Rate (FPR)  · False Negative Rate (FNR)
  Confusion Matrix  (TP / TN / FP / FN)

Output files
────────────
  evaluation/predictions.csv
      Account, GroundTruth, Probability, Prediction

Usage
─────
  python evaluation/evaluate.py
"""

import os
import sys
import csv
import pickle

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

############################################################
# Project Imports
############################################################

sys.path.append(".")

from representation.memory_projection import MemoryProjection
from representation.feature_fusion    import FeatureFusion
from fusion.fusion_network            import FusionNetwork

############################################################
# Paths
############################################################

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR  = os.path.join(PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR  = os.path.join(PROJECT_ROOT, "evaluation")

os.makedirs(EVAL_DIR, exist_ok=True)

FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_network.pth")
PREDICTIONS_PATH  = os.path.join(EVAL_DIR,  "predictions.csv")

############################################################
# Device
############################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2048

############################################################
# Classification Threshold
# ─────────────────────────────────────────────────────────
# Chosen via threshold_search.py to maximise F1.
# Change to 0.05 if Recall is the operating objective.
############################################################

THRESHOLD = 0.93

############################################################
# Load Artefacts
############################################################

print()
print("=" * 60)
print("VigilNet  —  Evaluation")
print("=" * 60)
print()

print("Loading Risk Memory …")
with open(os.path.join(DATA_DIR, "risk_memory.pkl"), "rb") as f:
    risk_memory = pickle.load(f)

print("Loading Behaviour Embeddings …")
with open(os.path.join(DATA_DIR, "behaviour_embeddings.pkl"), "rb") as f:
    behaviour_embeddings = pickle.load(f)

print("Loading Graph Embeddings …")
graph_embeddings = torch.load(
    os.path.join(DATA_DIR, "graph_embeddings.pt"),
    map_location="cpu"
)

print("Loading Transaction Sequences (labels) …")
with open(os.path.join(DATA_DIR, "transaction_sequences.pkl"), "rb") as f:
    transaction_sequences = pickle.load(f)

############################################################
# Canonical Account List
############################################################

accounts = sorted(transaction_sequences.keys())

############################################################
# Memory Feature Keys  (must match train_gat.py / train_fusion.py)
############################################################

MEMORY_FEATURE_KEYS = [
    "memory_score",
    "current_risk",
    "community_size",
    "community_density",
    "community_avg_risk",
    "community_max_risk",
]

############################################################
# Build Representation Modules
############################################################

memory_projection = MemoryProjection().to(DEVICE)
feature_fusion    = FeatureFusion().to(DEVICE)
memory_projection.eval()
feature_fusion.eval()

############################################################
# Pre-compute Local Embeddings
############################################################

print()
print("Building Local Embeddings …")

local_embeddings = []
labels_list      = []

with torch.no_grad():

    for account in accounts:

        # Memory vector
        account_memory = risk_memory.get(account, {})
        memory_vector  = torch.tensor(
            [float(account_memory.get(k, 0.0)) for k in MEMORY_FEATURE_KEYS],
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        # Behaviour vector
        behaviour_vector = torch.tensor(
            behaviour_embeddings.get(account, np.zeros(128, dtype=np.float32)),
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        memory_emb    = memory_projection(memory_vector)
        local_emb     = feature_fusion(memory_emb, behaviour_vector)

        local_embeddings.append(local_emb.squeeze(0).cpu())
        labels_list.append(transaction_sequences[account]["label"])

local_embeddings = torch.stack(local_embeddings)   # (N, 128)
labels           = torch.tensor(labels_list, dtype=torch.float32)

print(f"  Local embeddings : {local_embeddings.shape}")
print(f"  Graph embeddings : {graph_embeddings.shape}")
print(f"  Labels           : {labels.shape}")

############################################################
# Dataset & DataLoader
############################################################

class EvalDataset(Dataset):
    def __init__(self, local, graph, labels):
        self.local  = local
        self.graph  = graph
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.local[idx], self.graph[idx], self.labels[idx]


eval_dataset = EvalDataset(local_embeddings, graph_embeddings, labels)

eval_loader  = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

############################################################
# Load Trained Fusion Network
############################################################

print()
print(f"Loading model from  {FUSION_MODEL_PATH} …")

model = FusionNetwork().to(DEVICE)

checkpoint = torch.load(FUSION_MODEL_PATH, map_location=DEVICE)

# Support both raw state-dict and checkpoint dicts
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

############################################################
# Inference
############################################################

print("Running inference …")
print()

all_probs  = []
all_labels = []

with torch.no_grad():

    for local_emb, graph_emb, batch_labels in eval_loader:

        local_emb   = local_emb.to(DEVICE,   non_blocking=True)
        graph_emb   = graph_emb.to(DEVICE,   non_blocking=True)
        batch_labels = batch_labels.to(DEVICE, non_blocking=True)

        logits = model(local_emb, graph_emb)
        probs  = torch.sigmoid(logits).squeeze(1)

        all_probs.append(probs.cpu())
        all_labels.append(batch_labels.cpu())

all_probs  = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy().astype(int)
all_preds  = (all_probs >= THRESHOLD).astype(int)

############################################################
# Metrics
############################################################

accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds,    zero_division=0)
f1        = f1_score(all_labels, all_preds,        zero_division=0)

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

fpr     = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr     = fn / (fn + tp) if (fn + tp) > 0 else 0.0

roc_auc = roc_auc_score(all_labels, all_probs)
pr_auc  = average_precision_score(all_labels, all_probs)

############################################################
# Print Results
############################################################

W = 60

print("=" * W)
print(f"EVALUATION RESULTS  (threshold = {THRESHOLD})")
print("=" * W)
print()

print("Classification Metrics")
print("-" * W)
print(f"  {'Accuracy':<30}  {accuracy  * 100:>8.2f} %")
print(f"  {'Precision':<30}  {precision * 100:>8.2f} %")
print(f"  {'Recall':<30}  {recall    * 100:>8.2f} %")
print(f"  {'F1 Score':<30}  {f1        * 100:>8.2f} %")
print()

print("Ranking Metrics")
print("-" * W)
print(f"  {'ROC-AUC':<30}  {roc_auc:>9.4f}")
print(f"  {'PR-AUC':<30}  {pr_auc:>9.4f}")
print()

print("Error Rates")
print("-" * W)
print(f"  {'False Positive Rate (FPR)':<30}  {fpr * 100:>8.2f} %")
print(f"  {'False Negative Rate (FNR)':<30}  {fnr * 100:>8.2f} %")
print()

print("Confusion Matrix")
print("-" * W)
print(f"  {'True Positives  (TP)':<30}  {tp:>10,}")
print(f"  {'True Negatives  (TN)':<30}  {tn:>10,}")
print(f"  {'False Positives (FP)':<30}  {fp:>10,}")
print(f"  {'False Negatives (FN)':<30}  {fn:>10,}")
print()
print("=" * W)

############################################################
# Save Predictions CSV
############################################################

print()
print(f"Saving predictions to  {PREDICTIONS_PATH} …")

with open(PREDICTIONS_PATH, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    # Header
    writer.writerow(["Account", "GroundTruth", "Probability", "Prediction"])

    for account, gt, prob, pred in zip(
        accounts, all_labels, all_probs, all_preds
    ):
        writer.writerow([
            account,
            int(gt),
            f"{prob:.6f}",
            int(pred)
        ])

print(f"  Saved {len(accounts):,} rows.")
print()
print("Evaluation complete.")
print()
