import pickle
import torch
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from behaviour.transformer_encoder import BehaviourEncoder
from representation.memory_projection import MemoryProjection
from representation.feature_fusion import FeatureFusion
from gnn.gatv2_model import GATv2Model
from fusion.fusion_network import FusionNetwork

from online.transaction_processor import TransactionProcessor
from online.graph_updater import GraphUpdater
from online.risk_memory_updater import RiskMemoryUpdater


############################################################
# Device
############################################################

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)


############################################################
# Paths
############################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

DATA_DIR = PROJECT_ROOT / "data" / "processed"


############################################################
# Online AML System
############################################################

class OnlineAMLSystem:

    def __init__(self):

        ####################################################
        # Transaction Processor
        ####################################################

        self.transaction_processor = TransactionProcessor(

            DATA_DIR / "transaction_sequences.pkl"

        )

        ####################################################
        # Graph Updater
        ####################################################

        self.graph_updater = GraphUpdater(

            DATA_DIR / "weighted_graph.pkl"

        )

        ####################################################
        # Risk Memory
        ####################################################

        self.risk_updater = RiskMemoryUpdater(

            graph=self.graph_updater.graph,

            memory_file=DATA_DIR / "risk_memory.pkl"

        )

        ####################################################
        # Behaviour Encoder
        ####################################################

        self.behaviour_model = BehaviourEncoder(
            num_payment_formats=7,
            num_currencies=15
        )

        checkpoint = torch.load(

            MODEL_DIR / "behaviour_encoder.pth",

            map_location=DEVICE

        )

        self.behaviour_model.load_state_dict(

            checkpoint["encoder_state_dict"]

        )

        self.behaviour_model.to(DEVICE)

        self.behaviour_model.eval()

        ####################################################
        # Memory Projection
        ####################################################

        self.memory_projection = MemoryProjection()

        self.memory_projection.to(DEVICE)

        self.memory_projection.eval()

        ####################################################
        # Feature Fusion
        ####################################################

        self.feature_fusion = FeatureFusion()

        self.feature_fusion.to(DEVICE)

        self.feature_fusion.eval()

        ####################################################
        # GAT
        ####################################################

        self.gat = GATv2Model(
            hidden_dim=64,
            heads=2
        )

        checkpoint = torch.load(

            MODEL_DIR / "gatv2_model.pth",

            map_location=DEVICE

        )

        self.gat.load_state_dict(
            checkpoint["gat_state_dict"]
        )

        self.gat.to(DEVICE)

        self.gat.eval()
        
        ####################################################
        # Load Dataset and Embeddings for inference
        ####################################################
        
        from behaviour.behaviour_dataset import BehaviourDataset
        self.dataset = BehaviourDataset(DATA_DIR / "transaction_sequences.pkl")
        
        with open(DATA_DIR / "behaviour_embeddings.pkl", "rb") as f:
            self.behaviour_embeddings = pickle.load(f)

        self.graph_embeddings = torch.load(
            DATA_DIR / "graph_embeddings.pt",
            map_location=DEVICE
        )

        self.account_to_idx = self.dataset.account_to_index

        ####################################################
        # Fusion Network
        ####################################################

        self.fusion = FusionNetwork()

        checkpoint = torch.load(

            MODEL_DIR / "fusion_network.pth",

            map_location=DEVICE

        )

        self.fusion.load_state_dict(

            checkpoint["model_state_dict"]

        )

        self.fusion.to(DEVICE)

        self.fusion.eval()

    ########################################################
    # Predict
    ########################################################

    @torch.no_grad()

    def predict(

        self,

        sender,

        receiver,

        transaction

    ):

        ####################################################
        # Update Transaction History
        ####################################################

        self.transaction_processor.add_transaction(

            sender,

            receiver,

            transaction

        )

        ####################################################
        # Update Graph
        ####################################################

        self.graph_updater.add_transaction(

            sender=sender,

            receiver=receiver,

            amount=transaction["amount"],

            timestamp=transaction["timestamp"],

            payment_format=transaction["payment_format"],

            currency=transaction["currency"]

        )

        ####################################################
        # Update Risk Memory
        ####################################################

        memory = self.risk_updater.update(

            sender

        )

        ####################################################
        # Behaviour Sequence
        ####################################################

        sequence = self.transaction_processor.get_sequence(

            sender

        )

        ####################################################
        # Behaviour Preprocessing
        ####################################################

        account = sender

        idx = self.dataset.account_to_index[account]

        sequence_tensor, mask_tensor, _, _ = self.dataset[idx]

        sequence_tensor = sequence_tensor.unsqueeze(0).to(DEVICE)

        mask_tensor = mask_tensor.unsqueeze(0).to(DEVICE)

        ####################################################
        # Behaviour Encoder
        ####################################################

        behaviour_embedding, _ = self.behaviour_model(

            sequence_tensor,

            mask_tensor

        )

        ####################################################
        # Memory Projection
        ####################################################

        MEMORY_FEATURE_KEYS = [
            "memory_score", "current_risk", "community_size",
            "community_density", "community_avg_risk", "community_max_risk",
        ]

        if memory is None:
            memory = {}

        memory_values = [
            float(memory.get(k, 0.0))
            for k in MEMORY_FEATURE_KEYS
        ]

        memory_tensor = torch.tensor(

            memory_values,

            dtype=torch.float32,

            device=DEVICE

        ).unsqueeze(0)

        memory_embedding = self.memory_projection(

            memory_tensor

        )

        ####################################################
        # Local Representation
        ####################################################

        local_embedding = self.feature_fusion(

            memory_embedding,

            behaviour_embedding

        )

        ####################################################
        # Graph Embedding (precomputed)
        ####################################################

        idx = self.account_to_idx[sender]

        graph_embedding = self.graph_embeddings[idx].unsqueeze(0).to(DEVICE)

        ####################################################
        # Final Prediction
        ####################################################

        logit = self.fusion(

            local_embedding,

            graph_embedding

        )

        probability = torch.sigmoid(

            logit

        ).item()

        prediction = int(

            probability >= 0.93

        )

        return {

            "probability": probability,

            "prediction": prediction

        }


############################################################
# Testing
############################################################

if __name__ == "__main__":

    aml = OnlineAMLSystem()

    transaction = {

        "amount": 38769.39,

        "timestamp": 202209010021,

        "payment_format": "Cheque",

        "currency": "US Dollar",

        "time_gap": 3600,

        "in_degree": 14,

        "out_degree": 9

    }

    result = aml.predict(

        sender="121_8123FB9B0",

        receiver="10_8000EBD30",

        transaction=transaction

    )

    print()

    print("=" * 60)

    print("ONLINE AML RESULT")

    print("=" * 60)

    print()

    print(result)

    print()

    print("=" * 60)