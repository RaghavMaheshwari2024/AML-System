import torch
import torch.nn as nn


class MemoryProjection(nn.Module):
    """
    Projects handcrafted memory features into a learned embedding.

    Input:
        (batch_size, 6)

    Output:
        (batch_size, 64)
    """

    def __init__(
        self,
        input_dim=6,
        hidden_dim=32,
        output_dim=64,
        dropout=0.2
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.ReLU(),

            nn.BatchNorm1d(hidden_dim),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),

            nn.ReLU()

        )

    def forward(self, memory_vector):

        """
        memory_vector

        Shape:
            (batch_size, 6)
        """

        return self.network(memory_vector)


############################################################
# Testing
############################################################

if __name__ == "__main__":

    model = MemoryProjection()

    sample = torch.randn(8, 6)

    output = model(sample)

    print("Input Shape :", sample.shape)

    print("Output Shape:", output.shape)