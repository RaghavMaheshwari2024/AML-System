import math
from typing import cast
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 500):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # (1, max_len, embedding_dim)

        self.register_buffer("_pe", pe)

    @property
    def pe(self) -> torch.Tensor:
        return cast(torch.Tensor, self._pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}"
            )

        return x + self.pe[:, :seq_len]

if __name__ == "__main__":

    batch_size = 4
    seq_len = 50
    embedding_dim = 64

    x = torch.zeros(batch_size,
                    seq_len,
                    embedding_dim)

    pe = PositionalEncoding(
        embedding_dim=embedding_dim,
        max_len=100
    )

    output = pe(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)