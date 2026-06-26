"""
evaluation/threshold_search.py
────────────────────────────────────────────────────────────────────────────────
Three analysis steps, all driven from the saved predictions.csv:

  Step 1  Threshold Search
          Sweeps threshold ∈ [0.05, 0.95] in steps of 0.01.
          Reports the threshold that maximises F1 (and separately Recall).
          Prints a table of Precision / Recall / F1 at every threshold.

  Step 2  ROC and PR Curves
          Saves publication-quality figures to:
            evaluation/roc_curve.png
            evaluation/pr_curve.png

  Step 3  Error Analysis
          Reads predictions.csv, sorts by probability, and writes:
            evaluation/top100_false_positives.csv
            evaluation/top100_false_negatives.csv
          Also prints summary statistics for each group.

Usage
─────
  # Run evaluate.py first to produce predictions.csv, then:
  python evaluation/threshold_search.py
"""

import os
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

############################################################
# Paths
############################################################

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTIONS_CSV = os.path.join(EVAL_DIR, "predictions.csv")

ROC_PNG  = os.path.join(EVAL_DIR, "roc_curve.png")
PR_PNG   = os.path.join(EVAL_DIR, "pr_curve.png")
FP_CSV   = os.path.join(EVAL_DIR, "top100_false_positives.csv")
FN_CSV   = os.path.join(EVAL_DIR, "top100_false_negatives.csv")

############################################################
# Load predictions.csv
############################################################

print()
print("=" * 65)
print("VigilNet  —  Threshold Search & Error Analysis")
print("=" * 65)
print()

if not os.path.exists(PREDICTIONS_CSV):
    raise FileNotFoundError(
        f"predictions.csv not found at {PREDICTIONS_CSV}.\n"
        "Run  python evaluation/evaluate.py  first."
    )

print(f"Loading  {PREDICTIONS_CSV} …")

df = pd.read_csv(PREDICTIONS_CSV)

# Validate columns
required = {"Account", "GroundTruth", "Probability", "Prediction"}
if not required.issubset(df.columns):
    raise ValueError(f"predictions.csv must contain columns: {required}")

y_true  = df["GroundTruth"].values.astype(int)
y_prob  = df["Probability"].values.astype(float)

print(f"  Rows loaded      : {len(df):,}")
print(f"  Positive labels  : {y_true.sum():,}")
print(f"  Negative labels  : {(y_true == 0).sum():,}")
print()

############################################################
# ── STEP 1 : THRESHOLD SEARCH ────────────────────────────
############################################################

THRESHOLDS = np.arange(0.05, 0.96, 0.01)

records = []   # (threshold, precision, recall, f1)

for t in THRESHOLDS:
    preds = (y_prob >= t).astype(int)
    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true,    preds, zero_division=0)
    f = f1_score(y_true,        preds, zero_division=0)
    records.append((round(float(t), 2), p, r, f))

# Best by F1
best_f1_row   = max(records, key=lambda x: x[3])
# Best by Recall (tie-break: highest threshold → fewest FP)
best_recall_row = max(records, key=lambda x: (x[2], x[0]))

############################################################
# Print table
############################################################

header = f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>9}  {'F1':>8}"
sep    = "  " + "-" * 44

print("=" * 65)
print("THRESHOLD SWEEP")
print("=" * 65)
print(header)
print(sep)

for t, p, r, f in records:
    marker = ""
    if t == best_f1_row[0]:
        marker = "  ← best F1"
    if t == best_recall_row[0] and t != best_f1_row[0]:
        marker = "  ← best Recall"
    print(
        f"  {t:>10.2f}  {p * 100:>9.2f}%  {r * 100:>8.2f}%  {f * 100:>7.2f}%"
        f"{marker}"
    )

print(sep)
print()
print("Best by F1")
print("-" * 65)
t, p, r, f = best_f1_row
print(f"  Threshold  : {t:.2f}")
print(f"  Precision  : {p * 100:.2f} %")
print(f"  Recall     : {r * 100:.2f} %")
print(f"  F1 Score   : {f * 100:.2f} %")

print()
print("Best by Recall")
print("-" * 65)
t, p, r, f = best_recall_row
print(f"  Threshold  : {t:.2f}")
print(f"  Precision  : {p * 100:.2f} %")
print(f"  Recall     : {r * 100:.2f} %")
print(f"  F1 Score   : {f * 100:.2f} %")
print()

############################################################
# ── STEP 2 : ROC AND PR CURVES ───────────────────────────
############################################################

STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#a0a0b0",
    "ytick.color":       "#a0a0b0",
    "grid.color":        "#2a2d3d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "text.color":        "#e0e0e0",
    "font.family":       "sans-serif",
}
plt.rcParams.update(STYLE)

# ── ROC Curve ─────────────────────────────────────────────
fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true, y_prob)
roc_auc_val = auc(fpr_arr, tpr_arr)

# Mark optimal F1 threshold on ROC
opt_t = best_f1_row[0]
opt_idx = np.argmin(np.abs(roc_thresh - opt_t))

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    fpr_arr, tpr_arr,
    color="#7c6af7", linewidth=2.5, label=f"ROC  (AUC = {roc_auc_val:.4f})"
)
ax.plot([0, 1], [0, 1], color="#555566", linewidth=1.2, linestyle="--",
        label="Random classifier")
ax.scatter(
    fpr_arr[opt_idx], tpr_arr[opt_idx],
    color="#f7c948", s=90, zorder=5,
    label=f"Best-F1 threshold ({opt_t:.2f})"
)

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curve — Fusion Network", fontsize=14, pad=14)
ax.legend(fontsize=10, framealpha=0.3)
ax.grid(True)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

fig.tight_layout()
fig.savefig(ROC_PNG, dpi=150)
plt.close(fig)
print(f"Saved  {ROC_PNG}")

# ── PR Curve ──────────────────────────────────────────────
prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true, y_prob)
pr_auc_val = average_precision_score(y_true, y_prob)

# Baseline = prevalence
prevalence = y_true.mean()

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    rec_arr, prec_arr,
    color="#4fc97f", linewidth=2.5, label=f"PR  (AUC = {pr_auc_val:.4f})"
)
ax.axhline(
    prevalence, color="#555566", linewidth=1.2, linestyle="--",
    label=f"Baseline (prevalence = {prevalence * 100:.2f} %)"
)

# Mark best-F1 point
# find index in pr_thresh closest to opt_t
if len(pr_thresh) > 0:
    pr_opt_idx = np.argmin(np.abs(pr_thresh - opt_t))
    ax.scatter(
        rec_arr[pr_opt_idx], prec_arr[pr_opt_idx],
        color="#f7c948", s=90, zorder=5,
        label=f"Best-F1 threshold ({opt_t:.2f})"
    )

ax.set_xlabel("Recall",    fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve — Fusion Network", fontsize=14, pad=14)
ax.legend(fontsize=10, framealpha=0.3)
ax.grid(True)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

fig.tight_layout()
fig.savefig(PR_PNG, dpi=150)
plt.close(fig)
print(f"Saved  {PR_PNG}")
print()

############################################################
# ── STEP 3 : ERROR ANALYSIS ──────────────────────────────
############################################################

print("=" * 65)
print("ERROR ANALYSIS  (threshold = 0.50 from predictions.csv)")
print("=" * 65)
print()

# Use predictions as stored in CSV (threshold 0.50)
fp_mask = (df["Prediction"] == 1) & (df["GroundTruth"] == 0)
fn_mask = (df["Prediction"] == 0) & (df["GroundTruth"] == 1)

fp_df = df[fp_mask].sort_values("Probability", ascending=False).head(100)
fn_df = df[fn_mask].sort_values("Probability", ascending=True).head(100)

# ── Top 100 False Positives ───────────────────────────────
print("Top 100 False Positives  (highest model confidence, wrong)")
print(f"  Total FP in dataset : {fp_mask.sum():,}")
print(f"  Showing top 100 sorted by probability ↓")
print()
print(fp_df[["Account", "GroundTruth", "Probability", "Prediction"]]
      .head(10).to_string(index=False))
print(f"  … (see {os.path.basename(FP_CSV)} for full list)")
print()

fp_df.to_csv(FP_CSV, index=False)
print(f"Saved  {FP_CSV}")
print()

# ── Top 100 False Negatives ───────────────────────────────
print("Top 100 False Negatives  (lowest model confidence, missed positives)")
print(f"  Total FN in dataset : {fn_mask.sum():,}")
print(f"  Showing top 100 sorted by probability ↑  (model was least confident)")
print()
print(fn_df[["Account", "GroundTruth", "Probability", "Prediction"]]
      .head(10).to_string(index=False))
print(f"  … (see {os.path.basename(FN_CSV)} for full list)")
print()

fn_df.to_csv(FN_CSV, index=False)
print(f"Saved  {FN_CSV}")
print()

# ── Summary Stats ─────────────────────────────────────────
print("Probability Distribution Summary")
print("-" * 65)

groups = {
    "True  Positives (TP)": df[(df["Prediction"] == 1) & (df["GroundTruth"] == 1)]["Probability"],
    "True  Negatives (TN)": df[(df["Prediction"] == 0) & (df["GroundTruth"] == 0)]["Probability"],
    "False Positives (FP)": df[fp_mask]["Probability"],
    "False Negatives (FN)": df[fn_mask]["Probability"],
}

print(f"  {'Group':<25}  {'Count':>8}  {'Mean Prob':>10}  {'Median':>8}  {'Std':>8}")
print("  " + "-" * 64)

for name, series in groups.items():
    if len(series) == 0:
        print(f"  {name:<25}  {'0':>8}")
        continue
    print(
        f"  {name:<25}  {len(series):>8,}  "
        f"{series.mean():>10.4f}  {series.median():>8.4f}  {series.std():>8.4f}"
    )

print()
print("=" * 65)
print("Analysis complete.")
print()
print("Output files:")
print(f"  {ROC_PNG}")
print(f"  {PR_PNG}")
print(f"  {FP_CSV}")
print(f"  {FN_CSV}")
print()
