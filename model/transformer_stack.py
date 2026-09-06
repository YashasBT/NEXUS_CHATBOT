import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock


class TransformerStack(nn.Module):
    """
    Stack of multiple Transformer decoder blocks.

    Input:
        [B, T, C]

    Output:
        [B, T, C]

    Where:
        B = batch size
        T = sequence length
        C = embedding dimension
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        num_blocks,
        dropout=0.1,
        expansion_factor=4
    ):
        super().__init__()

        # -------------------------------------------------
        # Validate configuration
        # -------------------------------------------------

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
                "embedding_dim must be divisible by num_heads"
            )

        if context_length <= 0:
            raise ValueError(
                "context_length must be positive"
            )

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive"
            )

        # -------------------------------------------------
        # Create independent Transformer blocks
        # -------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    dropout=dropout,
                    expansion_factor=expansion_factor
                )
                for _ in range(num_blocks)
            ]
        )

        # -------------------------------------------------
        # Store configuration
        # -------------------------------------------------

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.expansion_factor = expansion_factor

    def forward(self, x):
        """
        Pass input through every Transformer block.

        Input:
            x -> [B, T, C]

        Output:
            [B, T, C]
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

        if x.size(1) > self.context_length:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds "
                f"context length {self.context_length}"
            )

        # -------------------------------------------------
        # Pass through Transformer blocks
        # -------------------------------------------------

        for block in self.blocks:
            x = block(x)

        return x


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NEXUS TRANSFORMER STACK TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # NEXUS configuration
    # -----------------------------------------------------

    batch_size = 2
    sequence_length = 16

    embedding_dim = 128
    num_heads = 4
    context_length = 128

    num_blocks = 4

    dropout = 0.1
    expansion_factor = 4

    # -----------------------------------------------------
    # Select device
    # -----------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print()
    print("Device:", device)

    # -----------------------------------------------------
    # GPU information
    # -----------------------------------------------------

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
    # Create Transformer Stack
    # -----------------------------------------------------

    stack = TransformerStack(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        context_length=context_length,
        num_blocks=num_blocks,
        dropout=dropout,
        expansion_factor=expansion_factor
    )

    # -----------------------------------------------------
    # Move model to GPU
    # -----------------------------------------------------

    stack = stack.to(device)

    # -----------------------------------------------------
    # Create input directly on same device
    # -----------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device
    )

    print()
    print("Input shape:", x.shape)
    print("Input device:", x.device)

    # -----------------------------------------------------
    # Verify model device
    # -----------------------------------------------------

    model_device = next(
        stack.parameters()
    ).device

    print(
        "Model device:",
        model_device
    )

    # -----------------------------------------------------
    # Device verification
    # -----------------------------------------------------

    assert model_device.type == device.type

    if device.type == "cuda":

        assert torch.cuda.is_available()

        assert model_device.type == "cuda"

    print(
        "Device verification: PASSED"
    )

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    output = stack(x)

    print()
    print("Output shape:", output.shape)
    print("Output device:", output.device)

    # -----------------------------------------------------
    # Verify output shape
    # -----------------------------------------------------

    assert output.shape == x.shape

    print(
        "Output shape verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify output device
    # -----------------------------------------------------

    assert output.device.type == x.device.type

    print(
        "Output device verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify number of blocks
    # -----------------------------------------------------

    print(
        "Number of Transformer blocks:",
        len(stack.blocks)
    )

    assert len(stack.blocks) == num_blocks

    print(
        "Block count verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify independent parameters
    # -----------------------------------------------------

    first_block_weight = (
        stack.blocks[0]
        .attention
        .query_projection
        .weight
    )

    second_block_weight = (
        stack.blocks[1]
        .attention
        .query_projection
        .weight
    )

    assert (
        first_block_weight
        is not second_block_weight
    )

    print(
        "Independent block parameters: PASSED"
    )

    # -----------------------------------------------------
    # Gradient test
    # -----------------------------------------------------

    loss = output.mean()

    print()
    print("Loss:", loss.item())

    loss.backward()

    print(
        "Gradient test: PASSED"
    )

    # -----------------------------------------------------
    # Verify at least one gradient exists
    # -----------------------------------------------------

    first_parameter = next(
        stack.parameters()
    )

    assert first_parameter.grad is not None

    print(
        "Gradient existence verification: PASSED"
    )

    # -----------------------------------------------------
    # Parameter count
    # -----------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in stack.parameters()
    )

    print()
    print(
        "Total Transformer Stack parameters:",
        f"{total_parameters:,}"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ALL TESTS PASSED.")
    print("=" * 60)