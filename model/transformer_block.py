import torch
import torch.nn as nn

from model.attention import MultiHeadSelfAttention
from model.feed_forward import FeedForward
from model.normalization import LayerNorm


class TransformerBlock(nn.Module):
    """
    A single Pre-Norm Transformer decoder block.

    Architecture:

        x
        │
        ├── LayerNorm
        │
        ├── Multi-Head Self-Attention
        │
        ├── Residual Connection
        │
        ├── LayerNorm
        │
        ├── Feed-Forward Network
        │
        └── Residual Connection

    Input:
        [B, T, C]

    Output:
        [B, T, C]
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1,
        expansion_factor=4
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible "
                "by num_heads"
            )

        # ---------------------------------------------
        # First LayerNorm
        # ---------------------------------------------

        self.norm1 = LayerNorm(
            embedding_dim=embedding_dim
        )

        # ---------------------------------------------
        # Multi-Head Self-Attention
        # ---------------------------------------------

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout
        )

        # ---------------------------------------------
        # Second LayerNorm
        # ---------------------------------------------

        self.norm2 = LayerNorm(
            embedding_dim=embedding_dim
        )

        # ---------------------------------------------
        # Feed-Forward Network
        # ---------------------------------------------

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout=dropout
        )

        # ---------------------------------------------
        # Dropout after sublayers
        # ---------------------------------------------

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through one Transformer block.

        Input:
            x -> [B, T, C]

        Output:
            [B, T, C]
        """

        if x.dim() != 3:
            raise ValueError(
                "Expected input shape [B, T, C]"
            )

        # ---------------------------------------------
        # Attention sub-block
        # ---------------------------------------------

        residual = x

        x = self.norm1(x)

        x = self.attention(x)

        x = self.dropout(x)

        x = residual + x

        # ---------------------------------------------
        # Feed-Forward sub-block
        # ---------------------------------------------

        residual = x

        x = self.norm2(x)

        x = self.feed_forward(x)

        x = self.dropout(x)

        x = residual + x

        return x


if __name__ == "__main__":

    # =================================================
    # NEXUS configuration
    # =================================================

    batch_size = 2
    sequence_length = 16

    embedding_dim = 128
    num_heads = 4
    context_length = 128

    dropout = 0.1
    expansion_factor = 4

    # =================================================
    # Device
    # =================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # =================================================
    # Create Transformer Block
    # =================================================

    block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        context_length=context_length,
        dropout=dropout,
        expansion_factor=expansion_factor
    ).to(device)

    # =================================================
    # Input
    # =================================================

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device
    )

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Input device:",
        x.device
    )

    # =================================================
    # Forward pass
    # =================================================

    output = block(x)

    print(
        "Output shape:",
        output.shape
    )

    print(
        "Output device:",
        output.device
    )

    # =================================================
    # Verify shape
    # =================================================

    assert output.shape == x.shape

    # =================================================
    # Verify CUDA placement
    # =================================================

    assert output.device == x.device

    # =================================================
    # Verify model parameters
    # =================================================

    model_device = next(
        block.parameters()
    ).device

    print(
        "Model device:",
        model_device
    )

    assert model_device == device

    # =================================================
    # Gradient test
    # =================================================

    loss = output.mean()

    loss.backward()

    print(
        "Gradient test: PASSED"
    )

    # =================================================
    # Final result
    # =================================================

    print()
    print("ALL TESTS PASSED.")