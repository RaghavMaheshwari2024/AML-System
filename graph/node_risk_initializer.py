import os
import pickle

# ==========================================================
# Configuration
# ==========================================================

INPUT_GRAPH = "data/processed/weighted_graph.pkl"
OUTPUT_FILE = "data/processed/node_risk_scores.pkl"

# ==========================================================
# Load Weighted Graph
# ==========================================================

print("Loading Weighted Graph...")

with open(INPUT_GRAPH, "rb") as f:
    G = pickle.load(f)

print("Graph Loaded!")

# ==========================================================
# Compute Initial Risk Scores
# ==========================================================

risk_scores = {}

for node in G.nodes():

    # Degree Information
    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)

    # Weighted Degree Information
    weighted_in = sum(
        data["weight"]
        for _, _, data in G.in_edges(node, data=True)
    )

    weighted_out = sum(
        data["weight"]
        for _, _, data in G.out_edges(node, data=True)
    )

    # Initial Risk Score
    risk = (
        in_degree +
        out_degree +
        weighted_in +
        weighted_out
    )

    risk_scores[node] = risk

# ==========================================================
# Normalize Scores to [0,1]
# ==========================================================

max_score = max(risk_scores.values())
min_score = min(risk_scores.values())

for node in risk_scores:

    if max_score == min_score:
        risk_scores[node] = 0.0
    else:
        risk_scores[node] = (
            risk_scores[node] - min_score
        ) / (max_score - min_score)

# ==========================================================
# Save Risk Scores
# ==========================================================

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(risk_scores, f)

print("\nNode Risk Initialization Completed!")

print(f"Total Nodes : {len(risk_scores):,}")

print(f"Saved to : {OUTPUT_FILE}")

# ==========================================================
# Print Top 10 Most Risky Nodes
# ==========================================================

print("\nTop 10 Initial Risk Nodes\n")

top_nodes = sorted(
    risk_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

for node, score in top_nodes:
    print(f"{node} : {score:.4f}")