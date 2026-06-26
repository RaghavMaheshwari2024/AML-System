import torch
import torch.nn as nn


class FusionNetwork(nn.Module):
    """
    Final AML Classifier

    Input:
        Local Representation (128)
        Graph Representation (128)

    Output:
        AML Logit (1)
    """

    def __init__(
        self,
        local_dim=128,
        graph_dim=128,
        hidden_dim=256,
        dropout=0.3
    ):

        super().__init__()

        ####################################################
        # Adaptive Fusion Gate
        ####################################################

        self.gate = nn.Sequential(

            nn.Linear(

                local_dim + graph_dim,

                128

            ),

            nn.ReLU(),

            nn.Linear(

            128,

            local_dim

            ),

            nn.Sigmoid()

        )

        ####################################################
        # Fusion Network
        ####################################################

        self.network = nn.Sequential(

            nn.Linear(

                local_dim + graph_dim,

                hidden_dim

            ),

            nn.BatchNorm1d(hidden_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            ################################################

            nn.Linear(

                hidden_dim,

                128

            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(dropout),

            ################################################

            nn.Linear(

                128,

                64

            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(dropout),

            ################################################

            nn.Linear(

                64,

                1

            )

        )

    ########################################################
    # Forward
    ########################################################
    def forward(

        self,

        local_embedding,

        graph_embedding

    ):

        ####################################################
        # Learn Adaptive Gate
        ####################################################

        gate = self.gate(

            torch.cat(

                [

                    local_embedding,

                    graph_embedding

                ],

                dim=1

            )

        )

        ####################################################
        # Adaptive Fusion
        ####################################################

        graph_embedding = gate * graph_embedding

        local_embedding = (1.0 - gate) * local_embedding

        ####################################################
        # Concatenate
        ####################################################

        fused = torch.cat(

            [

                local_embedding,

                graph_embedding

            ],

            dim=1

        )

        ####################################################
        # Classification
        ####################################################

        logits = self.network(

            fused

        )

        return logits


############################################################
# Testing
############################################################

if __name__ == "__main__":

    local = torch.randn(

        32,

        128

    )

    graph = torch.randn(

        32,

        128

    )

    model = FusionNetwork()

    output = model(

        local,

        graph

    )

    print()

    print("=" * 60)

    print("Fusion Network Test")

    print("=" * 60)

    print()

    print("Local Shape :", local.shape)

    print("Graph Shape :", graph.shape)

    print("Output Shape:", output.shape)

    print()

    print("=" * 60)