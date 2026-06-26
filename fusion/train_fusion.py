import os
import sys
import pickle
import random
import numpy as np

import torch
import torch.nn as nn

from torch.optim import AdamW

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

############################################################
# Import Project Modules
############################################################

sys.path.append(".")

from representation.memory_projection import MemoryProjection
from representation.feature_fusion import FeatureFusion

from fusion.fusion_network import FusionNetwork

############################################################
# Configuration
############################################################

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

BATCH_SIZE = 1024

EPOCHS = 50

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

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

MODEL_DIR = os.path.join(

    PROJECT_ROOT,

    "models"

)

os.makedirs(

    MODEL_DIR,

    exist_ok=True

)

MODEL_PATH = os.path.join(

    MODEL_DIR,

    "fusion_network.pth"

)

############################################################
# Load Risk Memory
############################################################

print()

print("Loading Risk Memory...")

with open(

    os.path.join(

        DATA_DIR,

        "risk_memory.pkl"

    ),

    "rb"

) as f:

    risk_memory = pickle.load(f)

############################################################
# Load Behaviour Embeddings
############################################################

print("Loading Behaviour Embeddings...")

with open(

    os.path.join(

        DATA_DIR,

        "behaviour_embeddings.pkl"

    ),

    "rb"

) as f:

    behaviour_embeddings = pickle.load(f)

############################################################
# Load Graph Embeddings
############################################################

print("Loading Graph Embeddings...")

graph_embeddings = torch.load(

    os.path.join(

        DATA_DIR,

        "graph_embeddings.pt"

    ),

    map_location="cpu"

)

############################################################
# Load Labels
############################################################

print("Loading Labels...")

with open(

    os.path.join(

        DATA_DIR,

        "transaction_sequences.pkl"

    ),

    "rb"

) as f:

    transaction_sequences = pickle.load(f)

############################################################
# Build Models
############################################################

memory_projection = MemoryProjection().to(

    DEVICE

)

feature_fusion = FeatureFusion().to(

    DEVICE

)

############################################################
# Freeze Representation Modules
############################################################

memory_projection.eval()

feature_fusion.eval()

for parameter in memory_projection.parameters():

    parameter.requires_grad = False

for parameter in feature_fusion.parameters():

    parameter.requires_grad = False

############################################################
# Account Ordering
############################################################

accounts = sorted(

    transaction_sequences.keys()

)

############################################################
# Memory Feature Keys
############################################################

MEMORY_FEATURE_KEYS = [

    "memory_score",

    "current_risk",

    "community_size",

    "community_density",

    "community_avg_risk",

    "community_max_risk"

]

############################################################
# Build Local Representations
############################################################

print()

print("Building Local Representations...")

local_embeddings = []

labels = []

with torch.no_grad():

    for account in accounts:

        ####################################################
        # Memory Vector
        ####################################################

        account_memory = risk_memory.get(

            account,

            {}

        )

        memory_vector = torch.tensor(

            [

                float(

                    account_memory.get(

                        key,

                        0.0

                    )

                )

                for key in MEMORY_FEATURE_KEYS

            ],

            dtype=torch.float32

        ).unsqueeze(0).to(

            DEVICE

        )

        ####################################################
        # Behaviour Vector
        ####################################################

        behaviour_vector = torch.tensor(

            behaviour_embeddings.get(

                account,

                np.zeros(

                    128,

                    dtype=np.float32

                )

            ),

            dtype=torch.float32

        ).unsqueeze(0).to(

            DEVICE

        )

        ####################################################
        # Memory Projection
        ####################################################

        memory_embedding = memory_projection(

            memory_vector

        )

        ####################################################
        # Feature Fusion
        ####################################################

        local_embedding = feature_fusion(

            memory_embedding,

            behaviour_vector

        )

        ####################################################
        # Store
        ####################################################

        local_embeddings.append(

            local_embedding.squeeze(0).cpu()

        )

        labels.append(

            transaction_sequences[account]["label"]

        )

############################################################
# Stack
############################################################

local_embeddings = torch.stack(

    local_embeddings

)

labels = torch.tensor(

    labels,

    dtype=torch.float32

)

print()

print("Local Embeddings")

print(local_embeddings.shape)

print()

print("Graph Embeddings")

print(graph_embeddings.shape)

print()

print("Labels")

print(labels.shape)


############################################################
# Fusion Dataset
############################################################

class FusionDataset(Dataset):

    def __init__(

        self,

        local_embeddings,

        graph_embeddings,

        labels

    ):

        self.local_embeddings = local_embeddings

        self.graph_embeddings = graph_embeddings

        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (

            self.local_embeddings[index],

            self.graph_embeddings[index],

            self.labels[index]

        )

############################################################
# Train / Validation Split
############################################################

num_samples = len(labels)

indices = torch.randperm(

    num_samples

)

train_size = int(

    0.8 * num_samples

)

train_indices = indices[:train_size]

val_indices = indices[train_size:]

############################################################
# Train Dataset
############################################################

train_dataset = FusionDataset(

    local_embeddings[train_indices],

    graph_embeddings[train_indices],

    labels[train_indices]

)

############################################################
# Validation Dataset
############################################################

val_dataset = FusionDataset(

    local_embeddings[val_indices],

    graph_embeddings[val_indices],

    labels[val_indices]

)

############################################################
# DataLoaders
############################################################

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=4,

    pin_memory=True

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4,

    pin_memory=True

)

############################################################
# Dataset Statistics
############################################################

positive = labels.sum().item()

negative = len(labels) - positive

print()

print("Dataset Statistics")

print("-" * 30)

print(f"Total Accounts    : {len(labels)}")

print(f"Positive Accounts : {int(positive)}")

print(f"Negative Accounts : {int(negative)}")

############################################################
# Positive Weight
############################################################

pos_weight = torch.tensor(

    [

        negative / max(

            positive,

            1

        )

    ],

    device=DEVICE

)

print()

print("Positive Weight")

print(pos_weight.item())

############################################################
# Build Model
############################################################

model = FusionNetwork().to(

    DEVICE

)

############################################################
# Loss Function
############################################################

criterion = nn.BCEWithLogitsLoss(

    pos_weight=pos_weight

)

############################################################
# Optimizer
############################################################

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

############################################################
# Scheduler
############################################################

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS

)

############################################################
# Setup Complete
############################################################

print()

print("=" * 60)

print("Setup Complete!")

print("=" * 60)

print()

print(model)

print()

print("Ready for Training...")


############################################################
# Metric Calculation
############################################################

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def calculate_metrics(labels, predictions):

    labels = labels.cpu().numpy()

    predictions = predictions.cpu().numpy()

    predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    return (

        accuracy,

        precision,

        recall,

        f1

    )


############################################################
# Train One Epoch
############################################################

def train_one_epoch():

    model.train()

    total_loss = 0

    all_labels = []

    all_predictions = []

    for (

        local_embedding,

        graph_embedding,

        labels

    ) in train_loader:

        local_embedding = local_embedding.to(

            DEVICE,

            non_blocking=True

        )

        graph_embedding = graph_embedding.to(

            DEVICE,

            non_blocking=True

        )

        labels = labels.to(

            DEVICE,

            non_blocking=True

        ).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(

            local_embedding,

            graph_embedding

        )

        loss = criterion(

            logits,

            labels

        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.sigmoid(

            logits

        )

        all_labels.append(

            labels.detach().cpu()

        )

        all_predictions.append(

            predictions.detach().cpu()

        )

    all_labels = torch.cat(

        all_labels

    )

    all_predictions = torch.cat(

        all_predictions

    )

    accuracy, precision, recall, f1 = calculate_metrics(

        all_labels,

        all_predictions

    )

    return (

        total_loss / len(train_loader),

        accuracy,

        precision,

        recall,

        f1

    )


############################################################
# Validation
############################################################

def validate():

    model.eval()

    total_loss = 0

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for (

            local_embedding,

            graph_embedding,

            labels

        ) in val_loader:

            local_embedding = local_embedding.to(

                DEVICE,

                non_blocking=True

            )

            graph_embedding = graph_embedding.to(

                DEVICE,

                non_blocking=True

            )

            labels = labels.to(

                DEVICE,

                non_blocking=True

            ).unsqueeze(1)

            logits = model(

                local_embedding,

                graph_embedding

            )

            loss = criterion(

                logits,

                labels

            )

            total_loss += loss.item()

            predictions = torch.sigmoid(

                logits

            )

            all_labels.append(

                labels.cpu()

            )

            all_predictions.append(

                predictions.cpu()

            )

    all_labels = torch.cat(

        all_labels

    )

    all_predictions = torch.cat(

        all_predictions

    )

    accuracy, precision, recall, f1 = calculate_metrics(

        all_labels,

        all_predictions

    )

    return (

        total_loss / len(val_loader),

        accuracy,

        precision,

        recall,

        f1

    )


############################################################
# Training Loop
############################################################

print()

print("Starting Training...")

best_f1 = 0.0

early_stop_counter = 0

EARLY_STOPPING_PATIENCE = 5

for epoch in range(EPOCHS):

    (

        train_loss,

        train_acc,

        train_precision,

        train_recall,

        train_f1

    ) = train_one_epoch()

    (

        val_loss,

        val_acc,

        val_precision,

        val_recall,

        val_f1

    ) = validate()

    scheduler.step()

    print("=" * 70)

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print()

    print("TRAIN")

    print(f"Loss      : {train_loss:.4f}")

    print(f"Accuracy  : {train_acc:.4f}")

    print(f"Precision : {train_precision:.4f}")

    print(f"Recall    : {train_recall:.4f}")

    print(f"F1 Score  : {train_f1:.4f}")

    print()

    print("VALIDATION")

    print(f"Loss      : {val_loss:.4f}")

    print(f"Accuracy  : {val_acc:.4f}")

    print(f"Precision : {val_precision:.4f}")

    print(f"Recall    : {val_recall:.4f}")

    print(f"F1 Score  : {val_f1:.4f}")

    if val_f1 > best_f1:

        best_f1 = val_f1

        early_stop_counter = 0

        checkpoint = {

            "model_state_dict":

                model.state_dict(),

            "optimizer_state_dict":

                optimizer.state_dict(),

            "epoch":

                epoch,

            "best_f1":

                best_f1

        }

        torch.save(

            checkpoint,

            MODEL_PATH

        )

        print()

        print("Best model saved!")

    else:

        early_stop_counter += 1

    if early_stop_counter >= EARLY_STOPPING_PATIENCE:

        print()

        print("Early stopping triggered.")

        break


############################################################
# Finished
############################################################

print()

print("=" * 70)

print("Training Complete!")

print()

print(f"Best Validation F1 : {best_f1:.4f}")

print()

print("Model saved to:")

print(MODEL_PATH)

print("=" * 70)