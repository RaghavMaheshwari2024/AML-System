explains the module, its inputs, outputs, assumptions, and artifacts ✅ 

                    OFFLINE INTELLIGENCE LAYER
────────────────────────────────────────────────────────────────────

IBM AML Dataset ✅ 
        │
        ▼
┌────────────────────────────┐
│ 1. Graph Builder           │  
└────────────────────────────┘
        │
        ▼
Transaction Graph (MultiDiGraph)  ✅ 
        │
        ▼
┌────────────────────────────┐
│ 2. AML Edge Weight Engine  │  
└────────────────────────────┘
        │
        ▼
Weighted Transaction Graph  ✅ 
        │
        ▼
┌────────────────────────────┐
│ 3. Node Risk Initializer   │  
└────────────────────────────┘
        │
        ▼
Initial Risk Scores  ✅ 
        │
        ▼
┌────────────────────────────┐
│ 4. AML Personalized        │
│    PageRank                │
└────────────────────────────┘
        │
        ▼
Global Risk Scores  ✅ 
        │
        ▼
┌────────────────────────────┐
│ 5. Community Intelligence  │
└────────────────────────────┘
        │
        ▼
Community Features  ✅ 
        │
        ▼
┌────────────────────────────┐
│ 6. Persistent Risk Memory  │
└────────────────────────────┘
        │
        ▼
Memory Vector per Account  ✅ 

────────────────────────────────────────────────────────────────────
REPRESENTATION LEARNING LAYER
────────────────────────────────────────────────────────────────────

Transaction History
        │
        ▼
┌────────────────────────────┐
│ 7. Behaviour Encoder       │
│    (Transformer Encoder)   │
└────────────────────────────┘
        │
        ▼
Behaviour Embeddings
              │
              │
              ▼

Memory Vector
      +
Behaviour Embedding
      │
      ▼
┌────────────────────────────┐
│ 8. Graph Attention Network │
│          (GATv2)           │
└────────────────────────────┘
        │
        ▼
Graph Embeddings

────────────────────────────────────────────────────────────────────
DECISION LAYER
────────────────────────────────────────────────────────────────────

Behaviour Embedding
        +
Graph Embedding
        │
        ▼
┌────────────────────────────┐
│ 9. Fusion Network          │
└────────────────────────────┘
        │
        ▼
Money Laundering Probability

Later, we can enrich the initial risk score with additional local signals from the IBM dataset, such as:

Number of unique counterparties
Incoming/outgoing amount ratio
Cross-bank transaction ratio
Currency diversity
Payment format diversity
Temporal burstiness (many transactions in a short period)

Those are all local features, so they belong here rather than in PageRank. By keeping the module modular, we can start with this simple implementation, verify the pipeline, and then iteratively make the initialization smarter without changing the downstream PageRank implementation.



graph.pkl
↓

weighted_graph.pkl
↓

node_risk_scores.pkl
↓

pagerank_scores.pkl
↓

community_features.pkl
↓

risk_memory.pkl