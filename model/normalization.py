import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization for Transformer representations.

    Input:
        [batch_size, sequence_length, embedding_dim]

    Output:
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.embedding_dim = embedding_dim

        self.norm = nn.LayerNorm(
            embedding_dim,
            eps=eps
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                "Expected input shape [B, T, C]"
            )

        if x.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"but got {x.size(-1)}"
            )

        return self.norm(x)


class ResidualConnection(nn.Module):
    """
    Adds the original input to a transformed output.

    output = x + sublayer_output
    """

    def forward(self, x, sublayer_output):
        if x.shape != sublayer_output.shape:
            raise ValueError(
                "Residual tensors must have identical shapes"
            )

        return x + sublayer_output


if __name__ == "__main__":

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    batch_size = 2
    sequence_length = 16
    embedding_dim = 128

    # -------------------------------------------------
    # CUDA device
    # -------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # -------------------------------------------------
    # Create LayerNorm
    # -------------------------------------------------

    layer_norm = LayerNorm(
        embedding_dim=embedding_dim
    ).to(device)

    # -------------------------------------------------
    # Create input
    # -------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device
    )

    print("Input shape:", x.shape)
    print("Input device:", x.device)

    # -------------------------------------------------
    # LayerNorm
    # -------------------------------------------------

    normalized = layer_norm(x)

    print(
        "Normalized shape:",
        normalized.shape
    )

    print(
        "Normalized device:",
        normalized.device
    )

    # -------------------------------------------------
    # Verify normalization
    # -------------------------------------------------

    means = normalized.mean(dim=-1)

    variances = normalized.var(
        dim=-1,
        unbiased=False
    )

    print(
        "Maximum absolute mean:",
        means.abs().max().item()
    )

    print(
        "Maximum variance error:",
        (variances - 1.0).abs().max().item()
    )

    # -------------------------------------------------
    # Residual connection
    # -------------------------------------------------

    residual = ResidualConnection().to(device)

    sublayer_output = torch.randn_like(x)

    output = residual(
        x,
        sublayer_output
    )

    print(
        "Residual output shape:",
        output.shape
    )

    print(
        "Residual output device:",
        output.device
    )

    # -------------------------------------------------
    # Assertions
    # -------------------------------------------------

    assert normalized.shape == x.shape
    assert normalized.device == x.device

    assert output.shape == x.shape
    assert output.device == x.device

    assert means.abs().max().item() < 1e-5

    assert (
        (variances - 1.0).abs().max().item()
        < 1e-4
    )

    assert torch.allclose(
        output,
        x + sublayer_output
    )

    print()
    print("ALL TESTS PASSED.")