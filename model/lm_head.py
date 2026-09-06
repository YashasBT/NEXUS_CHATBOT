import torch
import torch.nn as nn

from model.normalization import LayerNorm


class LanguageModelHead(nn.Module):
    """
    Final LayerNorm + vocabulary projection.

    Input:
        [B, T, C]

    Output:
        [B, T, V]

    Where:
        B = batch size
        T = sequence length
        C = embedding dimension
        V = vocabulary size
    """

    def __init__(
        self,
        embedding_dim,
        vocab_size
    ):
        super().__init__()

        # -------------------------------------------------
        # Validate configuration
        # -------------------------------------------------

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive"
            )

        # -------------------------------------------------
        # Final LayerNorm
        # -------------------------------------------------

        self.final_norm = LayerNorm(
            embedding_dim=embedding_dim
        )

        # -------------------------------------------------
        # Vocabulary projection
        # -------------------------------------------------

        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

        # -------------------------------------------------
        # Store configuration
        # -------------------------------------------------

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

    def forward(self, x):
        """
        Convert Transformer representations
        into vocabulary logits.

        Input:
            [B, T, C]

        Output:
            [B, T, V]
        """

        # -------------------------------------------------
        # Validate input
        # -------------------------------------------------

        if x.dim() != 3:
            raise ValueError(
                "Expected input shape [B, T, C]"
            )

        if x.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"but received {x.size(-1)}"
            )

        # -------------------------------------------------
        # Final normalization
        # -------------------------------------------------

        x = self.final_norm(x)

        # -------------------------------------------------
        # Project to vocabulary
        # -------------------------------------------------

        logits = self.output_projection(x)

        return logits


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NEXUS LANGUAGE MODEL HEAD TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    batch_size = 2
    sequence_length = 16

    embedding_dim = 128

    # Temporary test vocabulary.
    # Later this will come directly from the tokenizer.
    vocab_size = 50

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

    # -----------------------------------------------------
    # Create Language Model Head
    # -----------------------------------------------------

    lm_head = LanguageModelHead(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size
    ).to(device)

    # -----------------------------------------------------
    # Create Transformer output
    # -----------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device
    )

    print()
    print(
        "Transformer representation:",
        x.shape
    )

    print(
        "Input device:",
        x.device
    )

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    logits = lm_head(x)

    print()
    print(
        "Logits shape:",
        logits.shape
    )

    print(
        "Logits device:",
        logits.device
    )

    # -----------------------------------------------------
    # Expected shape
    # -----------------------------------------------------

    expected_shape = (
        batch_size,
        sequence_length,
        vocab_size
    )

    assert logits.shape == expected_shape

    print(
        "Logits shape verification: PASSED"
    )

    # -----------------------------------------------------
    # Device verification
    # -----------------------------------------------------

    assert logits.device.type == device.type

    print(
        "Device verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify vocabulary dimension
    # -----------------------------------------------------

    assert logits.size(-1) == vocab_size

    print(
        "Vocabulary dimension verification: PASSED"
    )

    # -----------------------------------------------------
    # Gradient test
    # -----------------------------------------------------

    loss = logits.mean()

    print()
    print(
        "Test loss:",
        loss.item()
    )

    loss.backward()

    print(
        "Gradient test: PASSED"
    )

    # -----------------------------------------------------
    # Verify gradients
    # -----------------------------------------------------

    projection_gradient = (
        lm_head
        .output_projection
        .weight
        .grad
    )

    assert projection_gradient is not None

    print(
        "Output projection gradient: PASSED"
    )

    # -----------------------------------------------------
    # Parameter count
    # -----------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in lm_head.parameters()
    )

    print()
    print(
        "Language Model Head parameters:",
        f"{total_parameters:,}"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ALL TESTS PASSED.")
    print("=" * 60)