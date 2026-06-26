# creates graph

import os
import pickle
import pandas as pd
import networkx as nx

# Load dataset
df = pd.read_csv("data/raw/HI-Small_Trans.csv")

# Create directed multigraph
G = nx.MultiDiGraph()

# Add transactions
for _, row in df.iterrows():

    source = f"{row['From Bank']}_{row['Account']}"
    target = f"{row['To Bank']}_{row['Account.1']}"

    G.add_edge(
        source,
        target,

        timestamp=row["Timestamp"],
        amount_paid=row["Amount Paid"],
        amount_received=row["Amount Received"],

        payment_currency=row["Payment Currency"],
        receiving_currency=row["Receiving Currency"],

        payment_format=row["Payment Format"],

        is_laundering=row["Is Laundering"]
    )

# Save graph
os.makedirs("data/processed", exist_ok=True)

with open("data/processed/graph.pkl", "wb") as f:
    pickle.dump(G, f)

print("Graph Created Successfully!")

print(f"Nodes : {G.number_of_nodes():,}")
print(f"Edges : {G.number_of_edges():,}")