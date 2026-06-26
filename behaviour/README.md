transaction_sequences.pkl
          │
          ▼
BehaviourDataset
          │
          ▼
50 × 6 Tensor
          │
          ▼
TransformerEncoder
          │
          ▼
128-D Behaviour Embedding
          │
          ▼
train_transformer.py
          │
          ▼
behaviour_embeddings.npy

--------------------------

Transformer Architecture

Input (50 × 6)
        │
        ▼
Continuous Projection (4 → 32)

Categorical Embeddings
    │          │
Payment     Currency
Embedding   Embedding
    │          │
    └────┬─────┘
         ▼
Concatenate
         │
         ▼
64-D Token Embedding
         │
         ▼
CLS Token
         │
         ▼
Positional Encoding
         │
         ▼
2 × Transformer Encoder Layers
         │
         ▼
CLS Output
         │
         ▼
Linear Layer
         │
         ▼
128-D Behaviour Embedding


----------------------------

HyperParameters

INPUT_FEATURES = 6

TOKEN_DIM = 64

NUM_HEADS = 4

NUM_LAYERS = 2

FEEDFORWARD_DIM = 256

DROPOUT = 0.1

OUTPUT_DIM = 128