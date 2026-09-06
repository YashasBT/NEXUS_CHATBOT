import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Input:
        [B, T, C]

    Output:
        [B, T, C]

    Where:

        B = batch size
        T = sequence length
        C = embedding dimension
        H = number of attention heads
        D = dimension of each attention head
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
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

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than zero."
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible "
                "by num_heads."
            )

        if context_length <= 0:
            raise ValueError(
                "context_length must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        # --------------------------------------------------
        # Store configuration
        # --------------------------------------------------

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.context_length = context_length

        # Head dimension:
        #
        # C = H * D
        #
        # Therefore:
        #
        # D = C / H

        self.head_dim = (
            embedding_dim // num_heads
        )

        # --------------------------------------------------
        # Q/K/V projection
        # --------------------------------------------------

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # --------------------------------------------------
        # Output projection
        # --------------------------------------------------

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # --------------------------------------------------
        # Dropout
        # --------------------------------------------------

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # Causal mask
        # --------------------------------------------------

        causal_mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
        )

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Split the embedding dimension into
        multiple attention heads.

        Input:
            [B, T, C]

        Output:
            [B, H, T, D]
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

    def _combine_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine multiple attention heads.

        Input:
            [B, H, T, D]

        Output:
            [B, T, C]
        """

        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2)

        x = x.contiguous()

        x = x.view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        return x

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Apply multi-head causal self-attention.

        Args:
            x:
                Input tensor [B, T, C].

            return_attention:
                Return attention weights when True.

        Returns:
            Output tensor [B, T, C].

            If return_attention=True:
                returns:
                    output,
                    attention_weights
        """

        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------

        if x.dim() != 3:
            raise ValueError(
                "Input must have shape [B, T, C]."
            )

        batch_size, sequence_length, embedding_dim = (
            x.shape
        )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                "Input embedding dimension does not "
                "match the configured embedding dimension."
            )

        if sequence_length > self.context_length:
            raise ValueError(
                "Sequence length cannot exceed "
                "context length."
            )

        # --------------------------------------------------
        # 1. Create Q, K, V
        # --------------------------------------------------

        queries = self.query_projection(x)

        keys = self.key_projection(x)

        values = self.value_projection(x)

        # Shapes:
        #
        # Q = [B, T, C]
        # K = [B, T, C]
        # V = [B, T, C]

        # --------------------------------------------------
        # 2. Split into heads
        # --------------------------------------------------

        queries = self._split_heads(queries)

        keys = self._split_heads(keys)

        values = self._split_heads(values)

        # Shapes:
        #
        # Q = [B, H, T, D]
        # K = [B, H, T, D]
        # V = [B, H, T, D]

        # --------------------------------------------------
        # 3. Calculate attention scores
        # --------------------------------------------------

        attention_scores = (
            queries
            @ keys.transpose(-2, -1)
        )

        # Shape:
        #
        # [B, H, T, D]
        #       ×
        # [B, H, D, T]
        #
        #       ↓
        #
        # [B, H, T, T]

        # --------------------------------------------------
        # 4. Scale scores
        # --------------------------------------------------

        attention_scores = (
            attention_scores
            / math.sqrt(self.head_dim)
        )

        # --------------------------------------------------
        # 5. Causal mask
        # --------------------------------------------------

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = (
            attention_scores.masked_fill(
                ~mask,
                torch.finfo(
                    attention_scores.dtype
                ).min,
            )
        )

        # --------------------------------------------------
        # 6. Softmax
        # --------------------------------------------------

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        # --------------------------------------------------
        # 7. Dropout on attention weights
        # --------------------------------------------------

        attention_weights = self.dropout(
            attention_weights
        )

        # --------------------------------------------------
        # 8. Weighted sum of values
        # --------------------------------------------------

        attention_output = (
            attention_weights @ values
        )

        # Shape:
        #
        # [B, H, T, T]
        #       ×
        # [B, H, T, D]
        #
        #       ↓
        #
        # [B, H, T, D]

        # --------------------------------------------------
        # 9. Combine heads
        # --------------------------------------------------

        attention_output = self._combine_heads(
            attention_output
        )

        # Shape:
        #
        # [B, T, C]

        # --------------------------------------------------
        # 10. Output projection
        # --------------------------------------------------

        output = self.output_projection(
            attention_output
        )

        # Shape:
        #
        # [B, T, C]

        if return_attention:
            return output, attention_weights

        return output


if __name__ == "__main__":
    # ==================================================
    # TEST CONFIGURATION
    # ==================================================

    embedding_dim = 128
    num_heads = 4
    context_length = 128
    dropout = 0.1

    batch_size = 2
    sequence_length = 16

    # ==================================================
    # CUDA
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ==================================================
    # CREATE MODEL
    # ==================================================

    attention = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        context_length=context_length,
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

    output, attention_weights = attention(
        x,
        return_attention=True,
    )

    # ==================================================
    # PRINT INFORMATION
    # ==================================================

    print("=" * 60)
    print("NEXUS MULTI-HEAD ATTENTION TEST")
    print("=" * 60)

    print("Device:")
    print(device)

    if device.type == "cuda":
        print("GPU:")
        print(
            torch.cuda.get_device_name(0)
        )

    print("\nConfiguration:")
    print(
        "Embedding dimension:",
        embedding_dim,
    )

    print(
        "Number of heads:",
        num_heads,
    )

    print(
        "Head dimension:",
        attention.head_dim,
    )

    print(
        "Context length:",
        context_length,
    )

    print("\nTensor shapes:")

    print(
        "Input:",
        x.shape,
    )

    # Inspect Q before splitting.
    q = attention.query_projection(x)

    print(
        "Q before splitting:",
        q.shape,
    )

    q_heads = attention._split_heads(q)

    print(
        "Q after splitting:",
        q_heads.shape,
    )

    print(
        "Attention weights:",
        attention_weights.shape,
    )

    print(
        "Output:",
        output.shape,
    )

    print("\nDevices:")

    print(
        "Input:",
        x.device,
    )

    print(
        "Model:",
        attention.query_projection.weight.device,
    )

    print(
        "Causal mask:",
        attention.causal_mask.device,
    )

    # ==================================================
    # EXPECTED SHAPES
    # ==================================================

    assert x.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    assert q_heads.shape == (
        batch_size,
        num_heads,
        sequence_length,
        attention.head_dim,
    )

    assert attention_weights.shape == (
        batch_size,
        num_heads,
        sequence_length,
        sequence_length,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    # ==================================================
    # CAUSAL TEST
    # ==================================================

    for row in range(sequence_length):

        for column in range(sequence_length):

            if column > row:

                future_attention = (
                    attention_weights[
                        :,
                        :,
                        row,
                        column,
                    ]
                )

                assert torch.all(
                    future_attention < 1e-6
                ), (
                    "Future token received "
                    "non-zero attention."
                )

    # ==================================================
    # DEVICE TEST
    # ==================================================

    assert (
        x.device
        == attention.query_projection.weight.device
    )

    assert (
        attention.causal_mask.device
        == x.device
    )

    # ==================================================
    # CONFIGURATION TEST
    # ==================================================

    assert (
        embedding_dim % num_heads == 0
    )

    assert (
        attention.head_dim
        == embedding_dim // num_heads
    )

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED.")
    print("=" * 60)