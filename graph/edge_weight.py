import os
import math
import pickle
import networkx as nx
from datetime import datetime

# ==========================================================
# Configuration
# ==========================================================

INPUT_GRAPH = "data/processed/graph.pkl"
OUTPUT_GRAPH = "data/processed/weighted_graph.pkl"

LAMBDA = 0.001

# ==========================================================
# Load Graph
# ==========================================================

print("Loading Graph...")

with open(INPUT_GRAPH, "rb") as f:
    G = pickle.load(f)

print("Graph Loaded!")

# ==========================================================
# Find Latest Timestamp
# ==========================================================

latest_timestamp = max(
    datetime.strptime(
        data["timestamp"], "%Y/%m/%d %H:%M"
    )
    for _, _, _, data in G.edges(keys=True, data=True)
)


# ==========================================================
# Aggregate Transactions
# ==========================================================

edge_stats = {}

for u, v, key, data in G.edges(keys=True, data=True):

    edge = (u, v)

    current_timestamp = datetime.strptime(
        data["timestamp"], "%Y/%m/%d %H:%M"
    )

    if edge not in edge_stats:

        edge_stats[edge] = {
            "frequency": 0,
            "weighted_amount": 0.0,
            "latest_time": current_timestamp
        }

    edge_stats[edge]["frequency"] += 1

    delta_t = latest_timestamp - current_timestamp
    delta_t_seconds = delta_t.total_seconds()

    edge_stats[edge]["weighted_amount"] += (

        math.log1p(data["amount_paid"])

        * math.exp(-LAMBDA * delta_t_seconds)

    )

    edge_stats[edge]["latest_time"] = max(
        edge_stats[edge]["latest_time"],
        current_timestamp
    )

# ==========================================================
# Compute Structural Overlap
# sigma = |N_out(u) ∩ N_in(v)|
# ==========================================================

WG = nx.DiGraph()

for (u, v), stats in edge_stats.items():

    out_neighbors = set(G.successors(u))

    in_neighbors = set(G.predecessors(v))

    sigma = len(out_neighbors.intersection(in_neighbors))

    structural_factor = 1 + math.log1p(sigma)

    frequency_factor = math.log1p(stats["frequency"])

    amount_factor = stats["weighted_amount"]

    weight = (
        frequency_factor
        * amount_factor
        * structural_factor
    )

    WG.add_edge(
        u,
        v,
        weight=weight,
        frequency=stats["frequency"],
        sigma=sigma,
        weighted_amount=amount_factor
    )

    for node in WG.nodes():

        total = sum(
            data["weight"]
            for _, _, data in WG.out_edges(node, data=True)
        )

        WG.nodes[node]["out_weight_sum"] = total

# ==========================================================
# Save Weighted Graph
# ==========================================================

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_GRAPH, "wb") as f:
    pickle.dump(WG, f)

print("\nWeighted Graph Created Successfully!")

print(f"Nodes : {WG.number_of_nodes():,}")
print(f"Edges : {WG.number_of_edges():,}")