import torch
import torch.nn as nn

from behaviour.positional_encoding import PositionalEncoding


class BehaviourEncoder(nn.Module):

    def __init__(
        self,
        num_payment_formats,
        num_currencies,
        continuous_dim=4,
        categorical_embedding_dim=32,
        token_dim=64,
        output_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_seq_length=50,
    ):

        super().__init__()

        # =====================================================
        # Continuous Feature Projection
        #
        # Continuous Features:
        # amount
        # direction
        # time_gap
        # counterparty_memory
        # =====================================================

        self.continuous_projection = nn.Linear(
            continuous_dim,
            categorical_embedding_dim
        )

        # =====================================================
        # Embeddings
        # =====================================================

        self.payment_embedding = nn.Embedding(
            num_payment_formats,
            categorical_embedding_dim
        )

        self.currency_embedding = nn.Embedding(
            num_currencies,
            categorical_embedding_dim
        )

        # =====================================================
        # Combine into token embedding
        # 32 + 32 + 32 = 96
        # =====================================================

        self.token_projection = nn.Linear(
            categorical_embedding_dim * 3,
            token_dim
        )

        # =====================================================
        # CLS Token
        # =====================================================

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, token_dim)
        )

        # =====================================================
        # Positional Encoding
        # =====================================================

        self.position_encoding = PositionalEncoding(
            embedding_dim=token_dim,
            max_len=max_seq_length + 1
        )

        # =====================================================
        # Transformer Encoder
        # =====================================================

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=token_dim,

            nhead=num_heads,

            dim_feedforward=feedforward_dim,

            dropout=dropout,

            activation="gelu",

            batch_first=True

        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )

        # =====================================================
        # Output Layer
        # =====================================================

        self.output_layer = nn.Sequential(

            nn.Linear(token_dim, token_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(token_dim, output_dim)

        )

    # =========================================================

    def forward(self, sequence, attention_mask):

        """
        sequence shape

        (batch, seq_len, 6)

        Feature order

        0 amount

        1 direction

        2 payment_format

        3 currency

        4 time_gap

        5 counterparty_memory
        """

        # -----------------------------------------------------
        # Continuous Features
        # -----------------------------------------------------

        continuous = torch.stack([

            sequence[:, :, 0],  # amount

            sequence[:, :, 1],  # direction

            sequence[:, :, 4],  # time_gap

            sequence[:, :, 5],  # counterparty_memory

        ], dim=-1)

        continuous = self.continuous_projection(
            continuous
        )

        # -----------------------------------------------------
        # Payment Embedding
        # -----------------------------------------------------

        payment = sequence[:, :, 2].long()

        payment = self.payment_embedding(
            payment
        )

        # -----------------------------------------------------
        # Currency Embedding
        # -----------------------------------------------------

        currency = sequence[:, :, 3].long()

        currency = self.currency_embedding(
            currency
        )

        # -----------------------------------------------------
        # Combine
        # -----------------------------------------------------

        tokens = torch.cat([

            continuous,

            payment,

            currency

        ], dim=-1)

        tokens = self.token_projection(tokens)

        # -----------------------------------------------------
        # CLS Token
        # -----------------------------------------------------

        batch_size = tokens.size(0)

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        tokens = torch.cat([

            cls,

            tokens

        ], dim=1)

        # -----------------------------------------------------
        # Attention Mask
        # True = ignore token
        # -----------------------------------------------------

        cls_mask = torch.ones(
            batch_size,
            1,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )

        attention_mask = torch.cat(

            [

                cls_mask,

                attention_mask

            ],

            dim=1

        )

        src_key_padding_mask = ~attention_mask

        # -----------------------------------------------------
        # Positional Encoding
        # -----------------------------------------------------

        tokens = self.position_encoding(
            tokens
        )

        # -----------------------------------------------------
        # Transformer
        # -----------------------------------------------------

        encoded = self.transformer(

            tokens,

            src_key_padding_mask=src_key_padding_mask

        )

        # -----------------------------------------------------
        # CLS Embedding
        # -----------------------------------------------------

        cls_embedding = encoded[:, 0]

        behaviour_embedding = self.output_layer(
            cls_embedding
        )

        return behaviour_embedding, encoded


# =============================================================
# Testing
# =============================================================

if __name__ == "__main__":

    model = BehaviourEncoder(

        num_payment_formats=10,

        num_currencies=10

    )

    sequence = torch.randn(8, 50, 6)

    sequence[:, :, 2] = torch.randint(
        0,
        10,
        (8, 50)
    )

    sequence[:, :, 3] = torch.randint(
        0,
        10,
        (8, 50)
    )

    mask = torch.ones(
        8,
        50,
        dtype=torch.bool
    )

    embedding, encoded = model(
        sequence,
        mask
    )

    print()

    print("Behaviour Embedding")

    print(embedding.shape)

    print()

    print("Encoded Sequence")

    print(encoded.shape)