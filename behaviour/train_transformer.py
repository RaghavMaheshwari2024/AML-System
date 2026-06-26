import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from behaviour_dataset import BehaviourDataset
from transformer_encoder import BehaviourEncoder
from pathlib import Path


############################################################
# Configuration
############################################################

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

TRAIN_RATIO = 0.8

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "behaviour_encoder.pth"


print("MODEL PATH:")
print(os.path.abspath(MODEL_PATH))

os.makedirs(MODEL_DIR, exist_ok=True)

############################################################
# Load Dataset
############################################################

print("Loading Dataset...")

dataset = BehaviourDataset()

print(f"Total Accounts : {len(dataset)}")

############################################################
# Train / Validation Split
############################################################

train_size = int(
    TRAIN_RATIO * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(

    dataset,

    [train_size, val_size]

)

############################################################
# DataLoader
############################################################

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=2

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2

)

############################################################
# Count Positive / Negative Accounts
############################################################

positive = 0

negative = 0

for _, _, label, _ in dataset:

    if label.item() == 1:

        positive += 1

    else:

        negative += 1

print()

print("Dataset Statistics")

print("-----------------------")

print("Positive Accounts :", positive)

print("Negative Accounts :", negative)

############################################################
# Weighted BCE Loss
############################################################

# Prevent division by zero

if positive == 0:

    positive = 1

pos_weight = torch.tensor(

    [negative / positive],

    dtype=torch.float32,

    device=DEVICE

)

print()

print("Positive Weight :", pos_weight.item())

############################################################
# Build Model
############################################################

model = BehaviourEncoder(

    num_payment_formats=len(
        dataset.payment_format_map
    ),

    num_currencies=len(
        dataset.currency_map
    )

)

model.to(DEVICE)

############################################################
# Classification Head
############################################################

classifier = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1)
)

classifier.to(DEVICE)

############################################################
# Loss Function
############################################################

criterion = nn.BCEWithLogitsLoss(

    pos_weight=pos_weight

)

############################################################
# Optimizer
############################################################

optimizer = torch.optim.AdamW(

    list(model.parameters())

    +

    list(classifier.parameters()),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

############################################################
# Learning Rate Scheduler
############################################################

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS

)

############################################################
# Best Validation Loss
############################################################

best_val_loss = float("inf")

print()

print("Setup Complete!")

print()

print(model)

print()

print(classifier)

print()

print("Ready for Training...")

############################################################
# Metrics
############################################################

def calculate_metrics(predictions, labels):

    predictions = predictions.int()
    labels = labels.int()

    TP = ((predictions == 1) & (labels == 1)).sum().item()
    TN = ((predictions == 0) & (labels == 0)).sum().item()
    FP = ((predictions == 1) & (labels == 0)).sum().item()
    FN = ((predictions == 0) & (labels == 1)).sum().item()

    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)

    precision = TP / max(TP + FP, 1)

    recall = TP / max(TP + FN, 1)

    f1 = (
        2 * precision * recall
        /
        max(precision + recall, 1e-8)
    )

    return accuracy, precision, recall, f1


############################################################
# Training Function
############################################################

def train_one_epoch():

    model.train()
    classifier.train()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for sequences, masks, labels, _ in train_loader:

        sequences = sequences.to(DEVICE)

        masks = masks.to(DEVICE)

        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        ####################################################
        # Forward Pass
        ####################################################

        embeddings, _ = model(

            sequences,

            masks

        )

        logits = classifier(

            embeddings

        )

        loss = criterion(

            logits,

            labels

        )

        ####################################################
        # Backpropagation
        ####################################################

        loss.backward()

        torch.nn.utils.clip_grad_norm_(

            list(model.parameters()) +
            list(classifier.parameters()),

            max_norm=1.0

        )

        optimizer.step()

        total_loss += loss.item()

        ####################################################
        # Predictions
        ####################################################

        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).float()

        all_predictions.append(

            preds.detach().cpu()

        )

        all_labels.append(

            labels.detach().cpu()

        )

    ########################################################

    all_predictions = torch.cat(

        all_predictions

    )

    all_labels = torch.cat(

        all_labels

    )

    accuracy, precision, recall, f1 = calculate_metrics(

        all_predictions,

        all_labels

    )

    average_loss = total_loss / len(train_loader)

    return (

        average_loss,

        accuracy,

        precision,

        recall,

        f1

    )


############################################################
# Validation Function
############################################################

def validate():

    model.eval()
    classifier.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for sequences, masks, labels, _ in val_loader:

            sequences = sequences.to(DEVICE)

            masks = masks.to(DEVICE)

            labels = labels.to(DEVICE).unsqueeze(1)

            ################################################

            embeddings, _ = model(

                sequences,

                masks

            )

            logits = classifier(

                embeddings

            )

            loss = criterion(

                logits,

                labels

            )

            total_loss += loss.item()

            ################################################

            probs = torch.sigmoid(logits)

            preds = (probs >= 0.5).float()

            all_predictions.append(

                preds.cpu()

            )

            all_labels.append(

                labels.cpu()

            )

    ########################################################

    all_predictions = torch.cat(

        all_predictions

    )

    all_labels = torch.cat(

        all_labels

    )

    accuracy, precision, recall, f1 = calculate_metrics(

        all_predictions,

        all_labels

    )

    average_loss = total_loss / len(val_loader)

    return (

        average_loss,

        accuracy,

        precision,

        recall,

        f1

    )

############################################################
# Training Loop
############################################################

print("\nStarting Training...\n")

best_f1 = 0.0

early_stop_counter = 0

EARLY_STOPPING_PATIENCE = 5

for epoch in range(EPOCHS):

    ########################################################
    # Train
    ########################################################

    (
        train_loss,
        train_acc,
        train_precision,
        train_recall,
        train_f1

    ) = train_one_epoch()

    ########################################################
    # Validation
    ########################################################

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1

    ) = validate()

    ########################################################
    # Scheduler
    ########################################################

    scheduler.step()

    ########################################################
    # Print Metrics
    ########################################################

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

    ########################################################
    # Save Best Model
    ########################################################

    if val_f1 > best_f1:

        best_f1 = val_f1

        early_stop_counter = 0

        checkpoint = {

            "encoder_state_dict":
                model.state_dict(),

            "classifier_state_dict":
                classifier.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "best_f1":
                best_f1,

            "payment_format_map":
                dataset.payment_format_map,

            "currency_map":
                dataset.currency_map

        }


        torch.save(

            checkpoint,

            MODEL_PATH

        )

        print()

        print("Best model saved!")

    else:

        early_stop_counter += 1

    ########################################################
    # Early Stopping
    ########################################################

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

print(f"Model saved to:")

print(MODEL_PATH)