"""
scripts/verify_alignment.py
────────────────────────────────────────────────────────────────────────────────
Verifies that all four data artefacts share the same canonical account list
and are perfectly aligned.

Checks
──────
  transaction_sequences.pkl   →  515 088 accounts (source of truth)
  behaviour_embeddings.pkl    →  same account count
  risk_memory.pkl             →  same account count
  graph_embeddings.pt         →  shape[0] == 515 088

Outputs
───────
  • Per-source count
  • First 10 accounts with presence flags
  • Missing-entry totals per source  (expected: 0 / 0 / 0)
"""

import os
import sys
import pickle

import numpy as np
import torch

############################################################
# Paths
############################################################

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

EXPECTED_COUNT = 515_088

############################################################
# Helper
############################################################

TICK  = "✓"
CROSS = "✗"

def _check(label: str, actual: int) -> None:
    status = TICK if actual == EXPECTED_COUNT else CROSS
    print(f"  {status}  {label:<35} {actual:>10,}")

############################################################
# Load Artefacts
############################################################

print()
print("=" * 60)
print("VigilNet  —  Data Alignment Verifier")
print("=" * 60)

print()
print("Loading artefacts …")
print()

# ── Transaction Sequences ─────────────────────────────────
with open(os.path.join(DATA_DIR, "transaction_sequences.pkl"), "rb") as f:
    transaction_sequences = pickle.load(f)

# ── Behaviour Embeddings ──────────────────────────────────
with open(os.path.join(DATA_DIR, "behaviour_embeddings.pkl"), "rb") as f:
    behaviour_embeddings = pickle.load(f)

# ── Risk Memory ───────────────────────────────────────────
with open(os.path.join(DATA_DIR, "risk_memory.pkl"), "rb") as f:
    risk_memory = pickle.load(f)

# ── Graph Embeddings ──────────────────────────────────────
graph_embeddings = torch.load(
    os.path.join(DATA_DIR, "graph_embeddings.pt"),
    map_location="cpu"
)

############################################################
# Canonical Account List
############################################################

accounts = sorted(transaction_sequences.keys())

############################################################
# Count Check
############################################################

print("Source Counts  (expected {:,})".format(EXPECTED_COUNT))
print("-" * 60)

_check("transaction_sequences  (accounts)", len(accounts))
_check("behaviour_embeddings   (entries)",  len(behaviour_embeddings))
_check("risk_memory            (entries)",  len(risk_memory))
_check("graph_embeddings       (rows)",     graph_embeddings.shape[0])

############################################################
# Spot-check: First 10 Accounts
############################################################

print()
print("First 10 Accounts — Presence Check")
print("-" * 60)
print(
    f"  {'#':<5}  {'Account ID':<20}  {'Behaviour':>10}  "
    f"{'Risk Mem':>9}  {'Graph Idx':>9}"
)
print("  " + "-" * 56)

for i, account in enumerate(accounts[:10]):

    has_behaviour = account in behaviour_embeddings
    has_memory    = account in risk_memory
    # graph embeddings are positional — index == sorted position
    graph_idx     = i

    print(
        f"  {i:<5}  {str(account):<20}  "
        f"{TICK if has_behaviour else CROSS:>10}  "
        f"{TICK if has_memory else CROSS:>9}  "
        f"{graph_idx:>9}"
    )

############################################################
# Missing Entry Counts
############################################################

print()
print("Missing Entry Report")
print("-" * 60)

account_set = set(accounts)

missing_behaviour = sum(
    1 for a in accounts if a not in behaviour_embeddings
)

missing_memory = sum(
    1 for a in accounts if a not in risk_memory
)

# Graph embeddings are positional so "missing" means row count mismatch
missing_graph = max(0, len(accounts) - graph_embeddings.shape[0])

def _miss(label: str, count: int) -> None:
    status = TICK if count == 0 else CROSS
    print(f"  {status}  {label:<35} {count:>10,}")

_miss("Missing behaviour embeddings",  missing_behaviour)
_miss("Missing risk memory entries",   missing_memory)
_miss("Missing graph embeddings",      missing_graph)

############################################################
# Final Verdict
############################################################

all_ok = (
    len(accounts)              == EXPECTED_COUNT and
    len(behaviour_embeddings)  == EXPECTED_COUNT and
    len(risk_memory)           == EXPECTED_COUNT and
    graph_embeddings.shape[0]  == EXPECTED_COUNT and
    missing_behaviour == 0     and
    missing_memory    == 0     and
    missing_graph     == 0
)

print()
print("=" * 60)

if all_ok:
    print(f"  {TICK}  ALL CHECKS PASSED — data is fully aligned.")
else:
    print(f"  {CROSS}  ALIGNMENT ISSUES DETECTED — see report above.")
    sys.exit(1)

print("=" * 60)
print()
