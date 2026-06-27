"""
scripts/generate_splits.py
────────────────────────────────────────────────────────────────────────────────
Creates a single, stratified, reproducible account-level split:

    70 % train  ·  15 % val  ·  15 % test

Output
──────
  data/processed/account_splits.pkl

  {
      "train": [account_id, ...],
      "val":   [account_id, ...],
      "test":  [account_id, ...],
  }

Usage
─────
  python scripts/generate_splits.py
"""

import os
import sys
import pickle

from pathlib import Path

from sklearn.model_selection import train_test_split

############################################################
# Paths
############################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEQUENCE_FILE = DATA_DIR / "transaction_sequences.pkl"

OUTPUT_FILE = DATA_DIR / "account_splits.pkl"

############################################################
# Configuration
############################################################

SEED = 42

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

############################################################
# Load Labels
############################################################

print()
print("=" * 60)
print("Generating Account Splits")
print("=" * 60)
print()

print("Loading transaction sequences …")

with open(SEQUENCE_FILE, "rb") as f:
    transaction_sequences = pickle.load(f)

accounts = sorted(transaction_sequences.keys())

labels = [
    transaction_sequences[account]["label"]
    for account in accounts
]

print(f"  Total accounts : {len(accounts)}")
print(f"  Positive       : {sum(labels)}")
print(f"  Negative       : {len(labels) - sum(labels)}")

############################################################
# Stratified Split
############################################################

print()
print("Creating stratified splits …")

# First split: train (70%) vs remaining (30%)
train_accounts, remaining_accounts, train_labels, remaining_labels = (
    train_test_split(
        accounts,
        labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=labels,
    )
)

# Second split: val (50% of remaining = 15%) vs test (50% of remaining = 15%)
val_accounts, test_accounts, val_labels, test_labels = (
    train_test_split(
        remaining_accounts,
        remaining_labels,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=remaining_labels,
    )
)

############################################################
# Print Statistics
############################################################

print()
print(f"  Train : {len(train_accounts):>8,}  ({len(train_accounts)/len(accounts)*100:.1f}%)  "
      f"positive={sum(train_labels)}")

print(f"  Val   : {len(val_accounts):>8,}  ({len(val_accounts)/len(accounts)*100:.1f}%)  "
      f"positive={sum(val_labels)}")

print(f"  Test  : {len(test_accounts):>8,}  ({len(test_accounts)/len(accounts)*100:.1f}%)  "
      f"positive={sum(test_labels)}")

############################################################
# Verify No Overlap
############################################################

train_set = set(train_accounts)
val_set = set(val_accounts)
test_set = set(test_accounts)

assert len(train_set & val_set) == 0, "Train/Val overlap!"
assert len(train_set & test_set) == 0, "Train/Test overlap!"
assert len(val_set & test_set) == 0, "Val/Test overlap!"
assert len(train_set | val_set | test_set) == len(accounts), "Missing accounts!"

print()
print("  ✓ No overlap between splits")
print("  ✓ All accounts assigned")

############################################################
# Save
############################################################

splits = {
    "train": train_accounts,
    "val": val_accounts,
    "test": test_accounts,
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(splits, f)

print()
print(f"Saved to: {OUTPUT_FILE}")
print()
print("=" * 60)
