import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    The same feed-forward network is applied
    independently to every token position.

    Input:
        [B, T, C]

    Output:
        [B, T, C]

    B = batch size
    T = sequence length
    C = embedding dimension
    """

    def __init__(
        self,
        embedding_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --------------------------------------------------
        # Validate configuration
        # --------------------------------------------------

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if expansion_factor <= 0:
            raise ValueError(
                "expansion_factor must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        # --------------------------------------------------
        # Store configuration
        # --------------------------------------------------

        self.embedding_dim = embedding_dim

        self.expansion_factor = expansion_factor

        self.hidden_dim = (
            embedding_dim * expansion_factor
        )

        # --------------------------------------------------
        # First linear transformation
        # --------------------------------------------------

        self.linear_in = nn.Linear(
            embedding_dim,
            self.hidden_dim,
        )

        # --------------------------------------------------
        # GELU activation
        # --------------------------------------------------

        self.gelu = nn.GELU()

        # --------------------------------------------------
        # Second linear transformation
        # --------------------------------------------------

        self.linear_out = nn.Linear(
            self.hidden_dim,
            embedding_dim,
        )

        # --------------------------------------------------
        # Dropout
        # --------------------------------------------------

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the feed-forward network.

        Args:
            x:
                Input tensor with shape [B, T, C].

        Returns:
            Tensor with shape [B, T, C].
        """

        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------

        if x.dim() != 3:
            raise ValueError(
                "Input must have shape [B, T, C]."
            )

        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                "Input embedding dimension does not "
                "match the configured embedding dimension."
            )

        # --------------------------------------------------
        # 1. Expand representation
        # --------------------------------------------------

        x = self.linear_in(x)

        # [B, T, C]
        #
        #      ↓
        #
        # [B, T, 4C]

        # --------------------------------------------------
        # 2. Apply nonlinear activation
        # --------------------------------------------------

        x = self.gelu(x)

        # [B, T, 4C]

        # --------------------------------------------------
        # 3. Project back to embedding dimension
        # --------------------------------------------------

        x = self.linear_out(x)

        # [B, T, 4C]
        #
        #      ↓
        #
        # [B, T, C]

        # --------------------------------------------------
        # 4. Apply dropout
        # --------------------------------------------------

        x = self.dropout(x)

        return x


if __name__ == "__main__":

    # ==================================================
    # CONFIGURATION
    # ==================================================

    embedding_dim = 128
    expansion_factor = 4
    dropout = 0.1

    batch_size = 2
    sequence_length = 16

    # ==================================================
    # CUDA DEVICE
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ==================================================
    # CREATE MODEL
    # ==================================================

    feed_forward = FeedForward(
        embedding_dim=embedding_dim,
        expansion_factor=expansion_factor,
        dropout=dropout,
    ).to(device)

    # ==================================================
    # CREATE INPUT
    # ==================================================

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device,
    )

    # ==================================================
    # FORWARD PASS
    # ==================================================

    output = feed_forward(x)

    # ==================================================
    # PRINT INFORMATION
    # ==================================================

    print("=" * 60)
    print("NEXUS FEED-FORWARD NETWORK TEST")
    print("=" * 60)

    print("\nDevice:")
    print(device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("\nConfiguration:")

    print(
        "Embedding dimension:",
        embedding_dim,
    )

    print(
        "Expansion factor:",
        expansion_factor,
    )

    print(
        "Hidden dimension:",
        feed_forward.hidden_dim,
    )

    print(
        "Dropout:",
        dropout,
    )

    print("\nTensor shapes:")

    print(
        "Input:",
        x.shape,
    )

    hidden = feed_forward.linear_in(x)

    print(
        "After Linear 1:",
        hidden.shape,
    )

    activated = feed_forward.gelu(hidden)

    print(
        "After GELU:",
        activated.shape,
    )

    projected = feed_forward.linear_out(
        activated
    )

    print(
        "After Linear 2:",
        projected.shape,
    )

    print(
        "Final output:",
        output.shape,
    )

    # ==================================================
    # DEVICE CHECK
    # ==================================================

    print("\nDevices:")

    print(
        "Input:",
        x.device,
    )

    print(
        "Linear layer:",
        feed_forward.linear_in.weight.device,
    )

    print(
        "Output:",
        output.device,
    )

    # ==================================================
    # SHAPE TESTS
    # ==================================================

    assert x.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    assert hidden.shape == (
        batch_size,
        sequence_length,
        embedding_dim * expansion_factor,
    )

    assert activated.shape == (
        batch_size,
        sequence_length,
        embedding_dim * expansion_factor,
    )

    assert projected.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    # ==================================================
    # DEVICE TEST
    # ==================================================

    assert (
        x.device
        == feed_forward.linear_in.weight.device
    )

    assert output.device == x.device

    print("\n" + "=" * 60)
    print("SHAPE TEST PASSED.")
    print("DEVICE TEST PASSED.")
    print("FEED-FORWARD TEST PASSED.")
    print("=" * 60)