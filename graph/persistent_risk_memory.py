import os
import pickle

# ==========================================================
# Configuration
# ==========================================================

PAGERANK_FILE = "data/processed/pagerank_scores.pkl"
COMMUNITY_FILE = "data/processed/community_features.pkl"
OUTPUT_FILE = "data/processed/risk_memory.pkl"

ALPHA = 0.7
BETA = 0.3

# ==========================================================
# Load Data
# ==========================================================

print("Loading PageRank Scores...")

with open(PAGERANK_FILE, "rb") as f:
    pagerank_scores = pickle.load(f)

print("Loading Community Features...")

with open(COMMUNITY_FILE, "rb") as f:
    community_features = pickle.load(f)

# ==========================================================
# Compute Persistent Memory
# ==========================================================

risk_memory = {}

for node in pagerank_scores:

    current_risk = pagerank_scores[node]

    community = community_features[node]

    community_risk = community["community_avg_risk"]

    memory_score = (
        ALPHA * current_risk
        +
        BETA * community_risk
    )

    risk_memory[node] = {

        "memory_score": memory_score,

        "current_risk": current_risk,

        "community_id": community["community_id"],

        "community_size": community["community_size"],

        "community_density": community["community_density"],

        "community_avg_risk": community["community_avg_risk"],

        "community_max_risk": community["community_max_risk"]

    }

# ==========================================================
# Save Memory
# ==========================================================

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(risk_memory, f)

print("\nPersistent Risk Memory Created!")

print(f"Total Nodes : {len(risk_memory):,}")

print(f"Saved to : {OUTPUT_FILE}")

# ==========================================================
# Display Top Memory Scores
# ==========================================================

print("\nTop 10 Memory Scores\n")

top_nodes = sorted(
    risk_memory.items(),
    key=lambda x: x[1]["memory_score"],
    reverse=True
)[:10]

for node, info in top_nodes:

    print(
        f"{node} : {info['memory_score']:.4f}"
    )