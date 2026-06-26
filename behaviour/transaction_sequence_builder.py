import os
import pickle
import pandas as pd
from collections import defaultdict

# ==========================================================
# Configuration
# ==========================================================

TRANSACTION_FILE = "data/raw/HI-Small_Trans.csv"
RISK_MEMORY_FILE = "data/processed/risk_memory.pkl"

OUTPUT_FILE = "data/processed/transaction_sequences.pkl"

MAX_SEQUENCE_LENGTH = 50

# ==========================================================
# Load Data
# ==========================================================

print("Loading Transactions...")

df = pd.read_csv(TRANSACTION_FILE)

print("Loading Risk Memory...")

with open(RISK_MEMORY_FILE, "rb") as f:
    risk_memory = pickle.load(f)

# ==========================================================
# Convert Timestamp
# ==========================================================

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# ==========================================================
# Sort Chronologically
# ==========================================================

df = df.sort_values("Timestamp")

# ==========================================================
# Build Sequences
# ==========================================================

account_labels = defaultdict(int)
account_sequences = defaultdict(list)

last_timestamp = {}

for _, row in df.iterrows():

    source = f"{row['From Bank']}_{row['Account']}"
    target = f"{row['To Bank']}_{row['Account.1']}"

    label = int(row["Is Laundering"])

    account_labels[source] = max(
        account_labels[source],
        label
    )

    account_labels[target] = max(
        account_labels[target],
        label
    )

    timestamp = row["Timestamp"]

    amount = float(row["Amount Paid"])

    payment_format = row["Payment Format"]

    currency = row["Payment Currency"]

    # ------------------------------------------------------
    # Source Account (Outgoing Transaction)
    # ------------------------------------------------------

    if source in last_timestamp:
        gap = (timestamp - last_timestamp[source]).total_seconds()
    else:
        gap = 0

    last_timestamp[source] = timestamp

    account_sequences[source].append({

        "amount": amount,

        "direction": 0,          # outgoing

        "payment_format": payment_format,

        "currency": currency,

        "time_gap": gap,

        "counterparty_memory":
            risk_memory.get(target, {}).get("memory_score", 0.0)

    })

    # ------------------------------------------------------
    # Target Account (Incoming Transaction)
    # ------------------------------------------------------

    if target in last_timestamp:
        gap = (timestamp - last_timestamp[target]).total_seconds()
    else:
        gap = 0

    last_timestamp[target] = timestamp

    account_sequences[target].append({

        "amount": float(row["Amount Received"]),

        "direction": 1,          # incoming

        "payment_format": payment_format,

        "currency": row["Receiving Currency"],

        "time_gap": gap,

        "counterparty_memory":
            risk_memory.get(source, {}).get("memory_score", 0.0)

    })

# ==========================================================
# Keep Last 50 Transactions
# ==========================================================

transaction_sequences = {}

for account, sequence in account_sequences.items():

    transaction_sequences[account] = {

        "transactions": sequence[-MAX_SEQUENCE_LENGTH:],

        "label": account_labels[account]

    }

# ==========================================================
# Save
# ==========================================================

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(transaction_sequences, f)

print("\nTransaction Sequences Created Successfully!")

print(f"Accounts : {len(transaction_sequences):,}")

print(f"Saved to : {OUTPUT_FILE}")

# ==========================================================
# Display Sample
# ==========================================================

sample_account = next(iter(transaction_sequences))

print("\nSample Account:\n")

print(sample_account)

print(transaction_sequences[sample_account]["transactions"][:3])