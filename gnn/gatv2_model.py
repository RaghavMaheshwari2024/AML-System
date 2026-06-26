import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv


class GATv2Model(nn.Module):

    """
    Baseline GATv2 Model

    Input:
        Node Features
        Edge Index
        Edge Weight

    Output:
        Graph Representation
    """

    def __init__(

        self,

        input_dim=128,

        hidden_dim=128,

        output_dim=128,

        heads=4,

        dropout=0.2

    ):

        super().__init__()

        ####################################################
        # Edge Encoder
        ####################################################

        self.edge_encoder = nn.Sequential(

            nn.Linear(1, 16),

            nn.ReLU(),

            nn.Linear(16, 16)

        )

        ####################################################
        # Hyperparameters
        ####################################################

        self.dropout = dropout

        self.heads = heads

        ####################################################
        # First GATv2 Layer
        ####################################################

        self.gat1 = GATv2Conv(

            input_dim,

            hidden_dim,

            heads=heads,

            dropout=dropout,

            edge_dim=16

        )

        ####################################################
        # BatchNorm
        ####################################################

        self.bn1 = nn.BatchNorm1d(

            hidden_dim * heads

        )

        ####################################################
        # Residual Projection
        ####################################################

        self.residual1 = nn.Linear(

            input_dim,

            hidden_dim * heads

        )

        ####################################################
        # Second GATv2 Layer
        ####################################################

        self.gat2 = GATv2Conv(

            hidden_dim*heads,

            output_dim,

            heads=1,

            concat=False,

            dropout=dropout,

            edge_dim=16

        )

        ####################################################
        # BatchNorm
        ####################################################

        self.bn2 = nn.BatchNorm1d(

            output_dim

        )

        ####################################################
        # Residual Projection
        ####################################################

        self.residual2 = nn.Linear(

            hidden_dim * heads,

            output_dim

        )

        ####################################################
        # Output Dropout
        ####################################################

        self.output_dropout = nn.Dropout(

            dropout

        )

        ####################################################
    # Forward
    ####################################################

    def forward(

        self,

        x,

        edge_index,

        edge_weight

    ):

        """
        Parameters
        ----------

        x :
            Node feature matrix

            Shape:
                (num_nodes, 128)

        edge_index :
            Graph connectivity

            Shape:
                (2, num_edges)

        edge_weight :
            AML edge weight

            Shape:
                (num_edges,)
        """

        ####################################################
        # Prepare Edge Features
        ####################################################

        if edge_weight.dim() == 1:

            edge_weight = edge_weight.unsqueeze(1)

        ####################################################
        # Save Residual
        ####################################################

        residual = self.residual1(x)
        edge_attr = self.edge_encoder(edge_weight)

        ####################################################
        # First GATv2 Layer
        ####################################################

        x = self.gat1(

            x,

            edge_index,

            edge_attr=edge_attr

        )

        x = self.bn1(x)

        x = F.relu(x)

        x = F.dropout(

            x,

            p=self.dropout,

            training=self.training

        )

        ####################################################
        # Residual Connection
        ####################################################

        x = x + residual

        ####################################################
        # Save Residual
        ####################################################

        residual = self.residual2(x)

        ####################################################
        # Second GATv2 Layer
        ####################################################

        x = self.gat2(

            x,

            edge_index,

            edge_attr=edge_attr

        )

        x = self.bn2(x)

        x = F.relu(x)

        x = self.output_dropout(x)

        ####################################################
        # Residual Connection
        ####################################################

        x = x + residual

        ####################################################
        # L2 Normalize Node Embeddings
        ####################################################

        x = F.normalize(

            x,

            p=2,

            dim=1

        )

        return x
    
    ############################################################
# Testing
############################################################

if __name__ == "__main__":

    from torch_geometric.data import Data

    ########################################################
    # Dummy Graph
    ########################################################

    num_nodes = 100

    num_edges = 500

    ########################################################
    # Random Node Features
    ########################################################

    x = torch.randn(

        num_nodes,

        128

    )

    ########################################################
    # Random Edge Index
    ########################################################

    edge_index = torch.randint(

        0,

        num_nodes,

        (2, num_edges)

    )

    ########################################################
    # Random AML Edge Weights
    ########################################################

    edge_weight = torch.rand(

        num_edges

    )

    ########################################################
    # Build Graph
    ########################################################

    graph = Data(

        x=x,

        edge_index=edge_index,

        edge_attr=edge_weight

    )

    ########################################################
    # Build Model
    ########################################################

    model = GATv2Model(

        input_dim=128,

        hidden_dim=128,

        output_dim=128,

        heads=4,

        dropout=0.2

    )

    ########################################################
    # Forward Pass
    ########################################################

    output = model(

        graph.x,

        graph.edge_index,

        graph.edge_attr

    )

    ########################################################
    # Results
    ########################################################

    print()

    print("=" * 60)

    print("GATv2 MODEL TEST")

    print("=" * 60)

    print()

    def _safe_shape(obj):
        try:
            return getattr(obj, 'shape', None)
        except Exception:
            return None

    print("Input Shape :")

    print(_safe_shape(graph.x))

    print()

    print("Output Shape :")

    print(_safe_shape(output))

    print()

    print("Edge Index Shape :")

    print(_safe_shape(graph.edge_index))

    print()

    print("Edge Weight Shape :")

    print(_safe_shape(graph.edge_attr))

    print()

    print("Forward Pass Successful!")

    print()

    print("=" * 60)