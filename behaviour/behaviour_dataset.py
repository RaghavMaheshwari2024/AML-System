import math
import pickle
import torch
from torch.utils.data import Dataset

# ==========================================================
# Configuration
# ==========================================================

SEQUENCE_FILE = "data/processed/transaction_sequences.pkl"

MAX_SEQ_LENGTH = 50

# ==========================================================
# Behaviour Dataset
# ==========================================================

class BehaviourDataset(Dataset):

    def __init__(self, sequence_file=SEQUENCE_FILE):

        with open(sequence_file, "rb") as f:
            self.sequences = pickle.load(f)

        self.accounts = list(self.sequences.keys())

        # Reverse mapping: account -> index
        self.account_to_index = {
            account: idx
            for idx, account in enumerate(self.accounts)
        }

        # Build categorical encodings
        self.payment_format_map = self._build_mapping("payment_format")
        self.currency_map = self._build_mapping("currency")

    # ======================================================

    def _build_mapping(self, feature):
        values = set()

        for account_data in self.sequences.values():
            for tx in account_data["transactions"]:
                values.add(tx[feature])

        values = sorted(list(values))

        return {
            value: idx
            for idx, value in enumerate(values)
        }

    # ======================================================

    def __len__(self):

        return len(self.accounts)

    # ======================================================

    def __getitem__(self, idx):

        account = self.accounts[idx]

        #sequence = self.sequences[account]
        account_data = self.sequences[account]

        sequence = account_data["transactions"]

        label = account_data["label"]

        tensor = []

        attention_mask = []

        for tx in sequence:

            amount = math.log1p(float(tx["amount"]))

            time_gap = math.log1p(float(tx["time_gap"]))

            tensor.append([

                amount,

                float(tx["direction"]),

                float(
                    self.payment_format_map[
                        tx["payment_format"]
                    ]
                ),

                float(
                    self.currency_map[
                        tx["currency"]
                    ]
                ),

                time_gap,

                float(tx["counterparty_memory"])

            ])

            attention_mask.append(1)

        # --------------------------------------------------
        # Padding
        # --------------------------------------------------

        while len(tensor) < MAX_SEQ_LENGTH:

            tensor.append([0.0] * 6)

            attention_mask.append(0)

        # Keep latest MAX_SEQ_LENGTH transactions

        tensor = tensor[-MAX_SEQ_LENGTH:]

        attention_mask = attention_mask[-MAX_SEQ_LENGTH:]

        tensor = torch.tensor(
            tensor,
            dtype=torch.float32
        )

        attention_mask = torch.tensor(
            attention_mask,
            dtype=torch.bool
        )

        #return tensor, attention_mask, account
        return (

            tensor,

            attention_mask,

            torch.tensor(
                label,
                dtype=torch.float32
            ),

            account
        )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    dataset = BehaviourDataset()

    print(f"Accounts : {len(dataset)}")

    sequence, mask, label, account = dataset[0]

    print("\nSample Account:")

    print(account)

    print("\nSequence Shape:")

    print(sequence.shape)

    print("\nAttention Mask Shape:")

    print(mask.shape)

    print("\nFirst Transaction:")

    print(sequence[0])

    print("\nAttention Mask:")

    print(mask)