What are offline modules?

Dataset
   │
   ▼
Graph Builder
   │
   ▼
AML Edge Weight Engine
   │
   ▼
Risk Diffusion Engine
   │
   ▼
Community Intelligence
   │
   ▼
Persistent Risk Memory
   │
   ▼
Graph Embedding Generator
   │
   ▼
Feature Store

-----------------------------
What are online modules?

Incoming Transaction
        │
        ▼
Transaction Feature Extractor
        │
        ▼
Recent Transaction Buffer
        │
        ▼
Feature Store Lookup
        │
        ▼
Behaviour Encoder
        │
        ▼
Fusion Network
        │
        ▼
Fraud Probability

-----------------------------
What artifacts does each module produce?

| Module             | Artifact                     | Mode
| ------------------ | ---------------------------- |
| Dataset Loader     | cleaned_transactions.parquet | ✓ Offline
| Graph Builder      | graph.pkl                    | ✓ Offline
| Edge Weight Engine | weighted_graph.pkl           | ✓ Offline
| Risk Diffusion     | pagerank_scores.parquet      | ✓ Offline
| Community Engine   | community_features.parquet   | ✓ Offline
| Persistent Memory  | risk_memory.parquet          | ✓ Offline
| Graph Embedding    | embeddings.npy               | ✓ Offline
| Feature Store      | redis dump                   | ✓ Offline

