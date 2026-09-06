import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense embedding vectors.

    Input:
        [B, T]

    Output:
        [B, T, C]

    B = batch size
    T = sequence length
    C = embedding dimension
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, token_ids):
        """
        Convert token IDs into embeddings.

        Input:
            [B, T]

        Output:
            [B, T, C]
        """

        if token_ids.dim() != 2:
            raise ValueError(
                "Expected token_ids shape [B, T]"
            )

        if token_ids.dtype != torch.long:
            raise ValueError(
                "token_ids must have dtype torch.long"
            )

        if token_ids.numel() > 0:

            if token_ids.min().item() < 0:
                raise ValueError(
                    "Token IDs cannot be negative"
                )

            if token_ids.max().item() >= self.vocab_size:
                raise ValueError(
                    "Token ID exceeds vocabulary size"
                )

        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    Input:
        sequence_length

    Output:
        [T, C]

    T = sequence length
    C = embedding dimension
    """

    def __init__(
        self,
        context_length,
        embedding_dim
    ):
        super().__init__()

        if context_length <= 0:
            raise ValueError(
                "context_length must be positive"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        self.context_length = context_length
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim
        )

    def forward(self, sequence_length):
        """
        Generate positional embeddings.

        Input:
            sequence_length -> integer

        Output:
            [T, C]
        """

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        # IMPORTANT:
        # Create position IDs on the same device
        # as the embedding weights.

        positions = torch.arange(
            sequence_length,
            device=self.embedding.weight.device
        )

        return self.embedding(positions)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NEXUS EMBEDDINGS TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    vocab_size = 50
    embedding_dim = 128
    context_length = 128

    batch_size = 2
    sequence_length = 16

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
    # Token Embedding
    # -----------------------------------------------------

    token_embedding = TokenEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    ).to(device)

    # -----------------------------------------------------
    # Positional Embedding
    # -----------------------------------------------------

    positional_embedding = PositionalEmbedding(
        context_length=context_length,
        embedding_dim=embedding_dim
    ).to(device)

    # -----------------------------------------------------
    # Create token IDs
    # -----------------------------------------------------

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length
        ),
        dtype=torch.long,
        device=device
    )

    print()
    print(
        "Token IDs shape:",
        token_ids.shape
    )

    print(
        "Token IDs device:",
        token_ids.device
    )

    # -----------------------------------------------------
    # Token embeddings
    # -----------------------------------------------------

    token_vectors = token_embedding(
        token_ids
    )

    print()
    print(
        "Token embeddings shape:",
        token_vectors.shape
    )

    print(
        "Token embeddings device:",
        token_vectors.device
    )

    # -----------------------------------------------------
    # Positional embeddings
    # -----------------------------------------------------

    position_vectors = positional_embedding(
        sequence_length
    )

    print()
    print(
        "Position embeddings shape:",
        position_vectors.shape
    )

    print(
        "Position embeddings device:",
        position_vectors.device
    )

    # -----------------------------------------------------
    # Combine embeddings
    # -----------------------------------------------------

    combined = (
        token_vectors
        + position_vectors
    )

    print()
    print(
        "Combined embeddings shape:",
        combined.shape
    )

    print(
        "Combined embeddings device:",
        combined.device
    )

    # -----------------------------------------------------
    # Verify shapes
    # -----------------------------------------------------

    assert token_vectors.shape == (
        batch_size,
        sequence_length,
        embedding_dim
    )

    assert position_vectors.shape == (
        sequence_length,
        embedding_dim
    )

    assert combined.shape == (
        batch_size,
        sequence_length,
        embedding_dim
    )

    print()
    print(
        "Shape verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify devices
    # -----------------------------------------------------

    assert token_vectors.device.type == device.type

    assert position_vectors.device.type == device.type

    assert combined.device.type == device.type

    print(
        "Device verification: PASSED"
    )

    # -----------------------------------------------------
    # Gradient test
    # -----------------------------------------------------

    loss = combined.mean()

    loss.backward()

    print(
        "Gradient test: PASSED"
    )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ALL TESTS PASSED.")
    print("=" * 60)