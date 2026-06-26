import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graph.aml_pagerank import AMLPageRank
from graph.community_engine import CommunityEngine
from graph.persistent_risk_memory import PersistentRiskMemory


class RiskMemoryUpdater:
    """
    Online updater for

        • AML PageRank
        • Community Statistics
        • Persistent Risk Memory

    No model retraining is performed.
    """

    ########################################################
    # Constructor
    ########################################################

    def __init__(

        self,

        graph,

        memory_file

    ):

        ####################################################
        # Graph
        ####################################################

        self.graph = graph

        ####################################################
        # Load Risk Memory
        ####################################################

        with open(memory_file, "rb") as f:

            self.memory = pickle.load(f)

        ####################################################
        # Engines
        ####################################################

        self.memory_engine = PersistentRiskMemory()

        ####################################################
        # Preload offline scores for fast online lookup
        ####################################################

        _data_dir = _ROOT / "data" / "processed"

        with open(_data_dir / "pagerank_scores.pkl", "rb") as f:
            self.risk_scores = pickle.load(f)

        with open(_data_dir / "community_features.pkl", "rb") as f:
            self.communities = pickle.load(f)

    ########################################################
    # Update
    ########################################################

    def update(

        self,

        account

    ):

        ####################################################
        # Update memory for this account using
        # precomputed offline PageRank + community scores
        ####################################################

        self.memory = self.memory_engine.update(

            memory=self.memory,

            risk_scores=self.risk_scores,

            communities=self.communities

        )

        return self.memory.get(

            account,

            None

        )

    ########################################################
    # Get Memory
    ########################################################

    def get_memory(

        self,

        account

    ):

        return self.memory.get(

            account,

            None

        )

    ########################################################
    # Save
    ########################################################

    def save(

        self,

        output_file

    ):

        with open(output_file, "wb") as f:

            pickle.dump(

                self.memory,

                f

            )


############################################################
# Testing
############################################################

if __name__ == "__main__":

    _graph_file = _ROOT / "data" / "processed" / "weighted_graph.pkl"
    _memory_file = _ROOT / "data" / "processed" / "risk_memory.pkl"

    with open(

        _graph_file,

        "rb"

    ) as f:

        graph = pickle.load(f)

    updater = RiskMemoryUpdater(

        graph=graph,

        memory_file=_memory_file

    )

    account = list(

        graph.nodes()

    )[0]

    updater.update(

        account

    )

    print()

    print("=" * 60)

    print("Updated Memory")

    print("=" * 60)

    print()

    print(

        updater.get_memory(

            account

        )

    )

    print()

    print("=" * 60)