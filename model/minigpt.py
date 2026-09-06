import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding, PositionalEmbedding
from model.transformer_stack import TransformerStack
from model.lm_head import LanguageModelHead


class MiniGPT(nn.Module):
    """
    Complete NEXUS GPT-style language model.

    Architecture:

        Token IDs
             ↓
        Token Embeddings
             +
        Positional Embeddings
             ↓
        Transformer Stack
             ↓
        Language Model Head
             ↓
        Vocabulary Logits

    Input:
        [B, T]

    Output:
        [B, T, V]

    B = batch size
    T = sequence length
    C = embedding dimension
    V = vocabulary size
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        context_length=128,
        num_heads=4,
        num_blocks=4,
        dropout=0.1,
        expansion_factor=4
    ):
        super().__init__()

        # -------------------------------------------------
        # Validate configuration
        # -------------------------------------------------

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if context_length <= 0:
            raise ValueError(
                "context_length must be positive"
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive"
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        # -------------------------------------------------
        # Store configuration
        # -------------------------------------------------

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.expansion_factor = expansion_factor

        # -------------------------------------------------
        # Token embedding
        # -------------------------------------------------

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        # -------------------------------------------------
        # Positional embedding
        # -------------------------------------------------

        self.position_embedding = PositionalEmbedding(
            context_length=context_length,
            embedding_dim=embedding_dim
        )

        # -------------------------------------------------
        # Transformer stack
        # -------------------------------------------------

        self.transformer_stack = TransformerStack(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            num_blocks=num_blocks,
            dropout=dropout,
            expansion_factor=expansion_factor
        )

        # -------------------------------------------------
        # Final LayerNorm + LM Head
        # -------------------------------------------------

        self.lm_head = LanguageModelHead(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size
        )

    def forward(self, token_ids):
        """
        Complete forward pass.

        Input:
            token_ids -> [B, T]

        Output:
            logits -> [B, T, V]
        """

        # -------------------------------------------------
        # Validate input
        # -------------------------------------------------

        if token_ids.dim() != 2:
            raise ValueError(
                "Expected token_ids shape [B, T]"
            )

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
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

        # -------------------------------------------------
        # Token embeddings
        # -------------------------------------------------

        token_embeddings = self.token_embedding(
            token_ids
        )

        # -------------------------------------------------
        # Positional embeddings
        # -------------------------------------------------

        position_embeddings = self.position_embedding(
            sequence_length
        )

        # -------------------------------------------------
        # Combine token + position information
        # -------------------------------------------------

        x = token_embeddings + position_embeddings

        # -------------------------------------------------
        # Transformer stack
        # -------------------------------------------------

        x = self.transformer_stack(x)

        # -------------------------------------------------
        # Language model head
        # -------------------------------------------------

        logits = self.lm_head(x)

        return logits


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 65)
    print("NEXUS COMPLETE MINIGPT TEST")
    print("=" * 65)

    # -----------------------------------------------------
    # NEXUS configuration
    # -----------------------------------------------------

    batch_size = 2
    sequence_length = 16

    vocab_size = 50

    embedding_dim = 128
    context_length = 128

    num_heads = 4
    num_blocks = 4

    dropout = 0.1
    expansion_factor = 4

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
    # Create model
    # -----------------------------------------------------

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
        expansion_factor=expansion_factor
    ).to(device)

    # -----------------------------------------------------
    # Create fake token IDs
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
    # Forward pass
    # -----------------------------------------------------

    logits = model(token_ids)

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
    # Expected output
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
    # Model device
    # -----------------------------------------------------

    model_device = next(
        model.parameters()
    ).device

    print(
        "Model device:",
        model_device
    )

    assert model_device.type == device.type

    print(
        "Model device verification: PASSED"
    )

    # -----------------------------------------------------
    # Parameter count
    # -----------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        "Total NEXUS parameters:",
        f"{total_parameters:,}"
    )

    # -----------------------------------------------------
    # Cross entropy test
    # -----------------------------------------------------

    # Create random next-token targets.
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length
        ),
        dtype=torch.long,
        device=device
    )

    # CrossEntropyLoss expects:
    #
    # logits:
    # [N, C]
    #
    # targets:
    # [N]
    #
    # Therefore flatten B and T.

    loss_function = nn.CrossEntropyLoss()

    loss = loss_function(
        logits.reshape(
            -1,
            vocab_size
        ),
        targets.reshape(-1)
    )

    print()
    print(
        "Cross-entropy loss:",
        loss.item()
    )

    assert loss.dim() == 0

    print(
        "Cross-entropy verification: PASSED"
    )

    # -----------------------------------------------------
    # Backpropagation
    # -----------------------------------------------------

    model.zero_grad()

    loss.backward()

    print(
        "Backward pass: PASSED"
    )

    # -----------------------------------------------------
    # Verify gradients
    # -----------------------------------------------------

    gradient_count = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            if parameter.grad is not None:
                gradient_count += 1

    print(
        "Parameters receiving gradients:",
        gradient_count
    )

    assert gradient_count > 0

    print(
        "Gradient verification: PASSED"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("=" * 65)
    print("ALL TESTS PASSED.")
    print("=" * 65)