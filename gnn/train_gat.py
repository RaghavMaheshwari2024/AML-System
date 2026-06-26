import os
import sys
import pickle
import random
import numpy as np

# Reduce CUDA memory fragmentation before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from torch.optim import AdamW

from torch_geometric.data import Data

############################################################
# Import Project Modules
############################################################

sys.path.append(".")

from representation.memory_projection import MemoryProjection
from representation.feature_fusion import FeatureFusion

from gnn.gatv2_model import GATv2Model

############################################################
# Configuration
############################################################

#DEVICE = torch.device("cpu")
DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

BATCH_SIZE = 512

# Mini-batch cluster size for ClusterData fallback
CLUSTER_PARTS = 150

EPOCHS = 50

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

# Chunked inference to avoid OOM during validate/embed-save
INFERENCE_CHUNK_SIZE = 50000

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

GAT_MODEL_PATH = os.path.join(

    MODEL_DIR,

    "gatv2_model.pth"

)

GRAPH_EMBEDDING_PATH = os.path.join(

    DATA_DIR,

    "graph_embeddings.pt"

)

os.makedirs(

    MODEL_DIR,

    exist_ok=True

)

############################################################
# Load Graph
############################################################

print()

print("Loading Graph...")

with open(

    os.path.join(

        DATA_DIR,

        "weighted_graph.pkl"

    ),

    "rb"

) as f:

    graph = pickle.load(f)

############################################################
# Load Risk Memory
############################################################

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
# Build Models
############################################################

memory_projection = MemoryProjection().to(

    DEVICE

)

feature_fusion = FeatureFusion().to(

    DEVICE

)

# Use projection and fusion modules in eval mode for feature extraction
memory_projection.eval()
feature_fusion.eval()

gat_model = GATv2Model(

    input_dim=128,

    hidden_dim=64,

    output_dim=128,

    heads=2,

    dropout=0.2

).to(DEVICE)

############################################################
# Convert Account IDs
############################################################

print()

print("Creating Node Mapping...")

accounts = sorted(

    graph.nodes()

)

account_to_index = {

    account: idx

    for idx, account

    in enumerate(accounts)

}

index_to_account = {

    idx: account

    for account, idx in account_to_index.items()

}

############################################################
# Build Node Features
############################################################

print("Building Node Features...")

MEMORY_FEATURE_KEYS = [
    "memory_score",
    "current_risk",
    "community_size",
    "community_density",
    "community_avg_risk",
    "community_max_risk",
]

node_features = []

for account in accounts:

    ########################################################

    account_memory = risk_memory.get(account, {})

    memory_vector = torch.tensor(

        [
            float(account_memory.get(key, 0.0))
            for key in MEMORY_FEATURE_KEYS
        ],

        dtype=torch.float32

    ).unsqueeze(0).to(DEVICE)

    ########################################################

    behaviour_vector = torch.tensor(

        behaviour_embeddings.get(account, np.zeros(128, dtype=np.float32)),

        dtype=torch.float32

    ).unsqueeze(0).to(DEVICE)

    ########################################################

    with torch.no_grad():

        memory_embedding = memory_projection(

            memory_vector

        )

        fused_feature = feature_fusion(

            memory_embedding,

            behaviour_vector

        )

    ########################################################

    node_features.append(

        fused_feature.squeeze(0).cpu()

    )

############################################################
# Stack Features
############################################################

node_features = torch.stack(

    node_features

)

print()

print("Node Feature Matrix")

print(node_features.shape)


############################################################
# Build Edge Index
############################################################

print()

print("Building Edge Index...")

edge_index = []

edge_weight = []

for u, v, data in graph.edges(data=True):

    edge_index.append([

        account_to_index[u],

        account_to_index[v]

    ])

    ########################################################
    # AML Edge Weight
    ########################################################

    edge_weight.append(

        float(data.get("weight", 1.0))

    )

############################################################
# Convert to Tensor
############################################################

edge_index = torch.tensor(

    edge_index,

    dtype=torch.long

).t().contiguous()

edge_weight = torch.tensor(

    edge_weight,

    dtype=torch.float32

)

print()

print("Edge Index Shape")

print(edge_index.shape)

print()

print("Edge Weight Shape")

print(edge_weight.shape)

############################################################
# Load Account Labels
############################################################

print()

print("Loading Account Labels...")

with open(

    os.path.join(

        DATA_DIR,

        "transaction_sequences.pkl"

    ),

    "rb"

) as f:

    transaction_sequences = pickle.load(f)

labels = []

for account in accounts:

    labels.append(

        transaction_sequences[account]["label"]

    )

labels = torch.tensor(

    labels,

    dtype=torch.float32

)

print()

print("Labels Shape")

print(labels.shape)

############################################################
# Build PyG Graph
############################################################

graph_data = Data(

    x=node_features,

    edge_index=edge_index,

    edge_attr=edge_weight.unsqueeze(1),

    y=labels

)

############################################################
# Train / Validation Split
############################################################

print()

print("Creating Train / Validation Split...")

############################################################
# Train / Validation Split
############################################################

num_nodes = graph_data.num_nodes

indices = torch.randperm(num_nodes)

train_size = int(0.8 * num_nodes)

train_indices = indices[:train_size]

val_indices = indices[train_size:]

train_mask = torch.zeros(num_nodes, dtype=torch.bool)

val_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_indices] = True

val_mask[val_indices] = True

graph_data.train_mask = train_mask

graph_data.val_mask = val_mask

############################################################
# Keep Graph on CPU — subgraphs are pushed to GPU per-batch
############################################################

# graph_data stays on CPU throughout; only mini-batches go to DEVICE
graph_data_cpu = graph_data  # alias for clarity

############################################################
# Manual Neighbor Sampler  (no pyg-lib / torch-sparse needed)
############################################################

class ManualNeighborSampler:
    """
    BFS-based neighborhood sampler built on numpy CSR adjacency.
    Requires only base torch + numpy — no pyg-lib or torch-sparse.
    Each __iter__ call yields a dict with subgraph tensors (on CPU);
    the training loop moves them to DEVICE individually.
    """

    def __init__(self, data, seed_mask, num_neighbors=(10, 5),
                 batch_size=512, shuffle=True):
        self.data          = data
        self.batch_size    = batch_size
        self.num_neighbors = num_neighbors
        self.shuffle       = shuffle
        self.seed_nodes    = seed_mask.nonzero(as_tuple=False).squeeze(1).numpy()

        n  = data.num_nodes
        ei = data.edge_index.numpy()          # (2, E)  — already on CPU
        dst = ei[1]

        # Build CSR: indptr[v] .. indptr[v+1] → incoming neighbours of v
        order             = np.argsort(dst, kind='stable')
        self._nbr_arr     = ei[0][order]                  # source nodes, sorted by dst
        self._indptr      = np.zeros(n + 1, dtype=np.int64)
        np.add.at(self._indptr, dst + 1, 1)
        np.cumsum(self._indptr, out=self._indptr)

        # Pre-cache edge arrays for fast subgraph extraction
        self._ei_src  = ei[0]
        self._ei_dst  = ei[1]
        self._ea      = data.edge_attr.numpy()   # (E, 1)
        self._x       = data.x.numpy()
        self._y       = data.y.numpy()
        self._n       = n

        # Reusable scratch buffers (avoids re-allocation)
        self._node_mask = np.zeros(n, dtype=bool)
        self._relabel   = np.full(n, -1, dtype=np.int64)

        print(f"ManualNeighborSampler ready — "
              f"{len(self.seed_nodes)} seeds, batch_size={batch_size}")

    def __len__(self):
        return max(1, (len(self.seed_nodes) + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        seeds = self.seed_nodes.copy()
        if self.shuffle:
            np.random.shuffle(seeds)
        for i in range(0, len(seeds), self.batch_size):
            yield self._build_batch(seeds[i: i + self.batch_size])

    def _build_batch(self, batch_seeds):
        # BFS neighbour expansion
        node_list = list(batch_seeds)
        node_set  = set(node_list)
        frontier  = node_list[:]

        for k in self.num_neighbors:
            new_frontier = []
            for node in frontier:
                s, e = self._indptr[node], self._indptr[node + 1]
                nbrs = self._nbr_arr[s:e]
                if len(nbrs) > k:
                    nbrs = np.random.choice(nbrs, k, replace=False)
                for nb in nbrs:
                    if nb not in node_set:
                        node_set.add(nb)
                        node_list.append(nb)
                        new_frontier.append(nb)
            frontier = new_frontier

        all_nodes = np.array(sorted(node_set), dtype=np.int64)
        n_sub     = len(all_nodes)

        # Extract induced subgraph (O(E) numpy ops)
        self._node_mask[all_nodes] = True
        edge_mask = self._node_mask[self._ei_src] & self._node_mask[self._ei_dst]
        sub_src   = self._ei_src[edge_mask]
        sub_dst   = self._ei_dst[edge_mask]
        sub_ea    = self._ea[edge_mask]

        # Relabel node indices 0 … n_sub-1
        self._relabel[all_nodes] = np.arange(n_sub, dtype=np.int64)
        sub_ei     = np.stack([self._relabel[sub_src], self._relabel[sub_dst]])
        seed_local = self._relabel[batch_seeds]   # local idx of seed nodes

        # Reset scratch buffers (only touched cells)
        self._node_mask[all_nodes] = False
        self._relabel[all_nodes]   = -1

        return {
            'x':          torch.from_numpy(self._x[all_nodes]).float(),
            'edge_index': torch.from_numpy(sub_ei).long(),
            'edge_attr':  torch.from_numpy(sub_ea).float(),
            'y':          torch.from_numpy(self._y[all_nodes]).float(),
            'seed_local': torch.from_numpy(seed_local).long(),
        }


############################################################
# Build Loaders
############################################################

use_full_graph_training = False
train_loader = None
val_loader   = None

try:
    from torch_geometric.loader import NeighborLoader
    from torch_geometric import typing as pyg_typing
    if not getattr(pyg_typing, 'WITH_TORCH_SPARSE', False):
        raise ImportError("torch_sparse/pyg-lib not available")

    train_loader = NeighborLoader(
        graph_data_cpu, input_nodes=graph_data_cpu.train_mask,
        num_neighbors=[10, 5], batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = NeighborLoader(
        graph_data_cpu, input_nodes=graph_data_cpu.val_mask,
        num_neighbors=[10, 5], batch_size=BATCH_SIZE, shuffle=False
    )
    print("Using NeighborLoader for mini-batch training.")

except Exception:
    print("pyg-lib/torch-sparse unavailable — using ManualNeighborSampler.")
    train_loader = ManualNeighborSampler(
        graph_data_cpu, graph_data_cpu.train_mask,
        num_neighbors=(10, 5), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = ManualNeighborSampler(
        graph_data_cpu, graph_data_cpu.val_mask,
        num_neighbors=(10, 5), batch_size=BATCH_SIZE, shuffle=False
    )

############################################################
# AMP scaler
############################################################

use_amp = (DEVICE.type == 'cuda')
scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)


############################################################
# Loss Function
############################################################

positive = labels.sum().item()

negative = len(labels) - positive

pos_weight = torch.tensor(

    [negative / max(positive, 1)],

    device=DEVICE

)

criterion = nn.BCEWithLogitsLoss(

    pos_weight=pos_weight

)

print()

print("=" * 60)

print("Graph Ready!")

print("=" * 60)

print()

print(graph_data)

############################################################
# Temporary Classification Head
############################################################

classifier = nn.Sequential(

    nn.Linear(128, 64),

    nn.ReLU(),

    nn.Dropout(0.2),

    nn.Linear(64, 1)

).to(DEVICE)

############################################################
# Optimizer
############################################################

optimizer = AdamW(

    list(gat_model.parameters()) +

    list(classifier.parameters()),

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
# Best Validation Loss
############################################################

best_f1 = 0.0

############################################################
# Ready
############################################################

print()

print("="*60)

print("Setup Complete!")

print("="*60)

print()

print(gat_model)

print()

print(classifier)

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

    accuracy = accuracy_score(labels, predictions)

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

    return accuracy, precision, recall, f1


############################################################
# Train One Epoch
############################################################

def _batch_to_device(batch):
    """Move a ManualNeighborSampler dict-batch or a PyG Data batch to DEVICE."""
    if isinstance(batch, dict):
        return {k: v.to(DEVICE) for k, v in batch.items()}
    return batch.to(DEVICE)


def train_one_epoch():

    gat_model.train()
    classifier.train()

    total_loss     = 0
    all_labels     = []
    all_predictions = []
    steps          = len(train_loader)

    for raw_batch in train_loader:

        batch = _batch_to_device(raw_batch)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):

            if isinstance(batch, dict):
                # ManualNeighborSampler dict batch
                embeddings  = gat_model(batch['x'], batch['edge_index'], batch['edge_attr'])
                seed_idx    = batch['seed_local']
                logits      = classifier(embeddings[seed_idx])
                batch_labels = batch['y'][seed_idx].unsqueeze(1)
            else:
                # PyG NeighborLoader batch
                embeddings  = gat_model(batch.x, batch.edge_index, batch.edge_attr)
                n           = batch.batch_size
                logits      = classifier(embeddings[:n])
                batch_labels = batch.y[:n].unsqueeze(1)

        if logits.numel() == 0:
            del batch, embeddings, logits, batch_labels
            torch.cuda.empty_cache()
            continue

        loss = criterion(logits, batch_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(gat_model.parameters()) + list(classifier.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_labels.append(batch_labels.detach().cpu())
        all_predictions.append(torch.sigmoid(logits).detach().cpu())

        del batch, embeddings, logits, batch_labels, loss
        torch.cuda.empty_cache()

    all_labels      = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_predictions)

    return total_loss / steps, accuracy, precision, recall, f1


def validate():
    """Mini-batch validation using val_loader (same sampler as training)."""
    gat_model.eval()
    classifier.eval()

    total_loss      = 0
    all_labels      = []
    all_predictions = []
    steps           = len(val_loader)

    with torch.no_grad():
        for raw_batch in val_loader:
            batch = _batch_to_device(raw_batch)

            with torch.amp.autocast('cuda', enabled=use_amp):
                if isinstance(batch, dict):
                    embeddings   = gat_model(batch['x'], batch['edge_index'], batch['edge_attr'])
                    seed_idx     = batch['seed_local']
                    logits       = classifier(embeddings[seed_idx])
                    batch_labels = batch['y'][seed_idx].unsqueeze(1)
                else:
                    embeddings   = gat_model(batch.x, batch.edge_index, batch.edge_attr)
                    n            = batch.batch_size
                    logits       = classifier(embeddings[:n])
                    batch_labels = batch.y[:n].unsqueeze(1)

            if logits.numel() == 0:
                continue

            loss = criterion(logits, batch_labels)
            total_loss += loss.item()
            all_labels.append(batch_labels.detach().cpu())
            all_predictions.append(torch.sigmoid(logits).detach().cpu())

            del batch, embeddings, logits, batch_labels, loss
            torch.cuda.empty_cache()

    all_labels      = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_predictions)

    return total_loss / steps, accuracy, precision, recall, f1

############################################################
# Training Loop
############################################################

print()

print("Starting Training...")

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
    # Print
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

            "gat_state_dict":

                gat_model.state_dict(),

            "classifier_state_dict":

                classifier.state_dict(),

            "optimizer_state_dict":

                optimizer.state_dict(),

            "epoch":

                epoch,

            "best_f1":

                best_f1

        }

        torch.save(

            checkpoint,

            GAT_MODEL_PATH

        )

        ####################################################
        # Save Graph Embeddings (mini-batch, no OOM)
        ####################################################

        gat_model.eval()

        # Simpler: run a dedicated embed pass using full node list in order
        x_cpu  = graph_data_cpu.x
        ei_cpu = graph_data_cpu.edge_index
        ea_cpu = graph_data_cpu.edge_attr
        EMBED_CHUNK = 20000   # nodes per chunk (subgraph is induced)
        node_emb_list = []

        with torch.no_grad():
            for start in range(0, graph_data_cpu.num_nodes, EMBED_CHUNK):
                end   = min(start + EMBED_CHUNK, graph_data_cpu.num_nodes)
                chunk = torch.arange(start, end)

                # Induced subgraph for this chunk
                chunk_np  = chunk.numpy()
                nm        = np.zeros(graph_data_cpu.num_nodes, dtype=bool)
                nm[chunk_np] = True
                ei_np     = ei_cpu.numpy()
                ea_np     = ea_cpu.numpy()
                emask     = nm[ei_np[0]] & nm[ei_np[1]]
                sub_src   = ei_np[0][emask]
                sub_dst   = ei_np[1][emask]
                rl        = np.full(graph_data_cpu.num_nodes, -1, dtype=np.int64)
                rl[chunk_np] = np.arange(len(chunk_np))
                sub_ei    = torch.from_numpy(
                    np.stack([rl[sub_src], rl[sub_dst]])
                ).long().to(DEVICE)
                sub_ea    = torch.from_numpy(ea_np[emask]).float().to(DEVICE)
                sub_x     = x_cpu[chunk].to(DEVICE)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    emb = gat_model(sub_x, sub_ei, sub_ea)

                node_emb_list.append(emb.cpu().float())

                del sub_x, sub_ei, sub_ea, emb
                torch.cuda.empty_cache()

        graph_embeddings = torch.cat(node_emb_list, dim=0)

        torch.save(

            graph_embeddings,

            GRAPH_EMBEDDING_PATH

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

print("Model Saved To")

print(GAT_MODEL_PATH)

print()

print("Embeddings Saved To")

print(GRAPH_EMBEDDING_PATH)

print("=" * 70)