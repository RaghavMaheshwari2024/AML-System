import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Adaptive gated fusion of

    Memory Embedding (64)

    and

    Behaviour Embedding (128)

    Output

    128-D Fused Embedding
    """

    def __init__(
        self,
        memory_dim=64,
        behaviour_dim=128,
        dropout=0.2
    ):

        super().__init__()

        ####################################################
        # Project Memory Embedding
        ####################################################

        self.memory_projection = nn.Sequential(

            nn.Linear(
                memory_dim,
                behaviour_dim
            ),

            nn.ReLU(),

            nn.Dropout(dropout)

        )

        ####################################################
        # Gate Network
        ####################################################

        self.gate = nn.Sequential(

            nn.Linear(

                behaviour_dim * 2,

                behaviour_dim

            ),

            nn.Sigmoid()

        )

    ########################################################

    def forward(

    self,

        memory_embedding,

        behaviour_embedding

    ):

        """
        memory_embedding
            (batch, 64)

        behaviour_embedding
            (batch, 128)
        """

        ####################################################
        # Project Memory
        ####################################################

        projected_memory = self.memory_projection(

        memory_embedding

        )

        ####################################################
        # Learnable Gate
        ####################################################

        gate = self.gate(

            torch.cat(

                [

                    projected_memory,

                    behaviour_embedding

                ],

                dim=1

            )

        )

    ####################################################
    # Adaptive Fusion
    ####################################################

        fused = (

            gate * projected_memory

            +

            (1.0 - gate) * behaviour_embedding

        )

        return fused


############################################################
# Testing
############################################################

if __name__ == "__main__":

    memory = torch.randn(16, 64)

    behaviour = torch.randn(16, 128)

    fusion = FeatureFusion()

    output = fusion(

        memory,

        behaviour

    )

    print()

    print("Memory Shape")

    print(memory.shape)

    print()

    print("Behaviour Shape")

    print(behaviour.shape)

    print()

    print("Fused Shape")

    print(output.shape)