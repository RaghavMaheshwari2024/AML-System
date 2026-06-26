import os
import pickle
import networkx as nx

# ==========================================================
# Configuration
# ==========================================================

GRAPH_FILE = "data/processed/weighted_graph.pkl"
RISK_FILE = "data/processed/node_risk_scores.pkl"
OUTPUT_FILE = "data/processed/pagerank_scores.pkl"

ALPHA = 0.85
MAX_ITER = 300
TOL = 1e-6

# ==========================================================
# Load Data
# ==========================================================

print("Loading Graph...")

with open(GRAPH_FILE, "rb") as f:
    G = pickle.load(f)

print("Loading Initial Risk Scores...")

with open(RISK_FILE, "rb") as f:
    personalization_raw = pickle.load(f)

nodes = list(G.nodes())
N = len(nodes)

print(f"Nodes : {N}")

if N == 0:
    raise ValueError("Graph has no nodes.")

# ==========================================================
# Normalize Personalization Vector
# ==========================================================

total = sum(personalization_raw.get(node, 0.0) for node in nodes)
if total == 0:
    raise ValueError("Personalization scores sum to zero.")

personalization = {
    node: personalization_raw.get(node, 0.0) / total
    for node in nodes
}

# ==========================================================
# Precompute Outgoing Weight Sums
# ==========================================================

out_weight_sum = {
    node: sum(G[node][nbr]["weight"] for nbr in G.successors(node))
    for node in nodes
}

# ==========================================================
# Initialize PageRank
# ==========================================================

pagerank = personalization.copy()

# ==========================================================
# Power Iteration
# ==========================================================

print("\nRunning Personalized PageRank...")

for iteration in range(MAX_ITER):
    # Handle dangling nodes: nodes with no outgoing weighted edges
    dangling_mass = sum(
        pagerank[node]
        for node in nodes
        if out_weight_sum[node] == 0
    )

    new_rank = {}

    for node in nodes:
        rank_sum = 0.0

        for predecessor in G.predecessors(node):
            total_weight = out_weight_sum[predecessor]
            if total_weight > 0:
                weight = G[predecessor][node]["weight"]
                rank_sum += pagerank[predecessor] * weight / total_weight

        # Redistribute dangling mass according to personalization
        new_rank[node] = (
            (1 - ALPHA) * personalization[node]
            + ALPHA * (rank_sum + dangling_mass * personalization[node])
        )

    # Convergence check: maximum node-wise change
    error = max(abs(new_rank[node] - pagerank[node]) for node in nodes)

    pagerank = new_rank

    print(f"Iteration {iteration + 1}  Error = {error:.15f}")

    if error < TOL:
        print("\nConverged!")
        break
else:
    print("\nWarning: did not converge within MAX_ITER.")

# ==========================================================
# Optional: Normalize Final Scores to [0, 1]
# ==========================================================

maximum = max(pagerank.values())
minimum = min(pagerank.values())

if maximum == minimum:
    pagerank = {node: 0.0 for node in pagerank}
else:
    pagerank = {
        node: (score - minimum) / (maximum - minimum)
        for node, score in pagerank.items()
    }

# ==========================================================
# Save Results
# ==========================================================

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(pagerank, f)

print("\nPageRank Scores Saved!")
print(f"Saved to : {OUTPUT_FILE}")

# ==========================================================
# Print Top 10 Risky Nodes
# ==========================================================

print("\nTop 10 Highest Risk Nodes\n")

top_nodes = sorted(
    pagerank.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

for node, score in top_nodes:
    print(f"{node} : {score:.4f}")