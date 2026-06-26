import pickle
import sys
from pathlib import Path

# Ensure the project root (AML-System/) is on sys.path so that
# sibling packages like `graph` are importable regardless of CWD.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graph.edge_weight import compute_edge_weight


class GraphUpdater:
    """
    Online graph updater.

    Responsibilities
    ----------------
    1. Add new transaction edge
    2. Update edge weight
    3. Maintain adjacency lists

    Does NOT perform:
        - PageRank
        - Community Detection
        - GNN Inference
    """

    def __init__(

        self,

        graph_file

    ):

        with open(graph_file, "rb") as f:

            self.graph = pickle.load(f)

    ########################################################
    # Add Transaction
    ########################################################

    def add_transaction(

        self,

        sender,

        receiver,

        amount,

        timestamp,

        payment_format,

        currency,

        **kwargs

    ):

        ####################################################
        # Ensure nodes exist
        ####################################################

        if not self.graph.has_node(sender):

            self.graph.add_node(sender)

        if not self.graph.has_node(receiver):

            self.graph.add_node(receiver)

        ####################################################
        # Compute Edge Weight
        ####################################################

        edge_weight = compute_edge_weight(

            amount=amount,

            timestamp=timestamp,

            **kwargs

        )

        ####################################################
        # Existing Edge — update attributes
        ####################################################

        if self.graph.has_edge(sender, receiver):

            attrs = self.graph[sender][receiver]

            attrs["count"]     += 1
            attrs["amount"]    += amount
            attrs["timestamp"]  = timestamp
            attrs["weight"]     = edge_weight

        ####################################################
        # New Edge
        ####################################################

        else:

            self.graph.add_edge(

                sender,
                receiver,
                amount=amount,
                count=1,
                timestamp=timestamp,
                payment_format=payment_format,
                currency=currency,
                weight=edge_weight

            )

    ########################################################
    # Get Neighbours
    ########################################################

    def get_neighbours(

        self,

        account

    ):

        if not self.graph.has_node(account):

            return []

        return list(

            self.graph.successors(account)

        )

    ########################################################
    # Get Edge Weight
    ########################################################

    def get_edge_weight(

        self,

        sender,

        receiver

    ):

        if not self.graph.has_node(sender):

            return None

        if not self.graph.has_edge(sender, receiver):

            return None

        return self.graph[sender][receiver].get("weight")

    ########################################################
    # Save
    ########################################################

    def save(

        self,

        output_file

    ):

        with open(output_file, "wb") as f:

            pickle.dump(

                self.graph,

                f

            )


############################################################
# Testing
############################################################

if __name__ == "__main__":

    _graph_file = _ROOT / "data" / "processed" / "weighted_graph.pkl"

    updater = GraphUpdater(_graph_file)

    updater.add_transaction(

        sender="ACC001",

        receiver="ACC002",

        amount=5000,

        timestamp=1720000000,

        payment_format="Wire",

        currency="USD"

    )

    print()

    print("=" * 60)

    print("Neighbours")

    print(

        updater.get_neighbours(

            "ACC001"

        )

    )

    print()

    print("Weight")

    print(

        updater.get_edge_weight(

            "ACC001",

            "ACC002"

        )

    )

    print()

    print("=" * 60)