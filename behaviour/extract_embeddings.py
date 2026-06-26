import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from behaviour_dataset import BehaviourDataset
from transformer_encoder import BehaviourEncoder

# ==========================================================
# Configuration
# ==========================================================

with open("data/processed/behaviour_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print(len(embeddings))

sample = next(iter(embeddings))

print(sample)

print(embeddings[sample].shape)


with open("data/processed/risk_memory.pkl","rb") as f:
    memory = pickle.load(f)

print(len(memory))


print(sample in memory)

MODEL_PATH = "models/behaviour_encoder.pth"

OUTPUT_NUMPY = "data/processed/behaviour_embeddings.npy"

OUTPUT_DICT = "data/processed/behaviour_embeddings.pkl"

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# Load Dataset
# ==========================================================

dataset = BehaviourDataset()

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ==========================================================
# Load Model
# ==========================================================

model = BehaviourEncoder(

    num_payment_formats=len(dataset.payment_format_map),

    num_currencies=len(dataset.currency_map)

)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["encoder_state_dict"]
)

model.to(DEVICE)

model.eval()

# ==========================================================
# Extract Embeddings
# ==========================================================

all_embeddings = []

embedding_dict = {}

print("Extracting Behaviour Embeddings...")

with torch.no_grad():

    for sequences, masks, labels,  accounts in loader:

        sequences = sequences.to(DEVICE)

        masks = masks.to(DEVICE)

        embeddings, _ = model(
            sequences,
            masks
        )

        embeddings = embeddings.cpu().numpy()

        all_embeddings.append(embeddings)

        for account, embedding in zip(accounts, embeddings):

            embedding_dict[account] = embedding

# ==========================================================
# Save Results
# ==========================================================

all_embeddings = np.concatenate(

    all_embeddings,

    axis=0

)

os.makedirs(

    "data/processed",

    exist_ok=True

)

np.save(

    OUTPUT_NUMPY,

    all_embeddings

)

with open(

    OUTPUT_DICT,

    "wb"

) as f:

    pickle.dump(

        embedding_dict,

        f

    )

print()

print("Behaviour Embeddings Saved!")

print()

print("Embedding Shape")

print(all_embeddings.shape)

print()

print("Total Accounts")

print(len(embedding_dict))

print()

sample = next(iter(embedding_dict))

print("Sample Account")

print(sample)

print()

print("Embedding Dimension")

print(len(embedding_dict[sample]))