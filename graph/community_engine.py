import os
import pickle
import networkx as nx
import community as community_louvain

# ==========================================================
# Configuration
# ==========================================================

GRAPH_FILE = "data/processed/weighted_graph.pkl"
PAGERANK_FILE = "data/processed/pagerank_scores.pkl"
OUTPUT_FILE = "data/processed/community_features.pkl"

# ==========================================================
# Load Data
# ==========================================================

print("Loading Weighted Graph...")

with open(GRAPH_FILE, "rb") as f:
    G = pickle.load(f)

print("Loading PageRank Scores...")

with open(PAGERANK_FILE, "rb") as f:
    pagerank = pickle.load(f)

# ==========================================================
# Convert to Undirected Graph
# (Louvain works on undirected graphs)
# ==========================================================

UG = G.to_undirected()

print("Detecting Communities...")

partition = community_louvain.best_partition(UG, weight="weight")

print(f"Communities Found : {len(set(partition.values()))}")

# ==========================================================
# Compute Community Statistics
# ==========================================================

community_features = {}

communities = {}

for node, cid in partition.items():

    communities.setdefault(cid, []).append(node)

for cid, nodes in communities.items():

    subgraph = UG.subgraph(nodes)

    size = len(nodes)

    density = nx.density(subgraph)

    risks = [pagerank.get(node,0.0) for node in nodes]

    avg_risk = sum(risks) / len(risks)

    max_risk = max(risks)

    for node in nodes:

        community_features[node] = {

            "community_id": cid,

            "community_size": size,

            "community_density": density,

            "community_avg_risk": avg_risk,

            "community_max_risk": max_risk

        }

# ==========================================================
# Save Community Features
# ==========================================================

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(community_features, f)

print("\nCommunity Features Saved!")

print(f"Total Nodes : {len(community_features)}")

print(f"Saved to : {OUTPUT_FILE}")

# ==========================================================
# Print Largest Communities
# ==========================================================

print("\nLargest Communities:\n")

largest = sorted(
    communities.items(),
    key=lambda x: len(x[1]),
    reverse=True
)[:10]

for cid, nodes in largest:

    print(f"Community {cid} : {len(nodes)} nodes")