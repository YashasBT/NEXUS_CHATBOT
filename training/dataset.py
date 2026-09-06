import torch
from torch.utils.data import Dataset, DataLoader


class NextTokenDataset(Dataset):
    """
    Dataset for autoregressive next-token prediction.

    Given a token sequence:

        [t0, t1, t2, t3, t4, ...]

    and a context length C, each sample contains:

        Input:
            [ti, ti+1, ..., ti+C-1]

        Target:
            [ti+1, ti+2, ..., ti+C]

    Input shape:
        [C]

    Target shape:
        [C]
    """

    def __init__(
        self,
        token_ids,
        context_length
    ):
        super().__init__()

        if context_length <= 0:
            raise ValueError(
                "context_length must be positive"
            )

        # Convert incoming token IDs into a tensor.
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(
                token_ids,
                dtype=torch.long
            )
        else:
            token_ids = token_ids.to(
                dtype=torch.long
            )

        # We need context_length + 1 tokens
        # to create one input/target pair.
        if len(token_ids) <= context_length:
            raise ValueError(
                "Not enough tokens to create a "
                "next-token prediction sample"
            )

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):
        """
        Number of possible sliding-window samples.
        """

        return (
            len(self.token_ids)
            - self.context_length
        )

    def __getitem__(self, index):
        """
        Return one input/target pair.
        """

        start = index
        end = index + self.context_length

        # Input:
        # [t0, t1, ..., t(C-1)]
        x = self.token_ids[
            start:end
        ]

        # Target:
        # [t1, t2, ..., tC]
        y = self.token_ids[
            start + 1:end + 1
        ]

        return x, y


def create_dataloader(
    token_ids,
    context_length,
    batch_size,
    shuffle=True,
    drop_last=True
):
    """
    Create a DataLoader for next-token prediction.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    dataset = NextTokenDataset(
        token_ids=token_ids,
        context_length=context_length
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    return loader


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 65)
    print("NEXUS NEXT-TOKEN DATASET TEST")
    print("=" * 65)

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    context_length = 8
    batch_size = 4

    # -----------------------------------------------------
    # Fake token stream for testing
    #
    # In the real training pipeline this will come
    # from your tokenizer and training.txt.
    # -----------------------------------------------------

    token_ids = torch.arange(
        30,
        dtype=torch.long
    )

    print()
    print(
        "Total tokens:",
        len(token_ids)
    )

    print(
        "Context length:",
        context_length
    )

    # -----------------------------------------------------
    # Create dataset
    # -----------------------------------------------------

    dataset = NextTokenDataset(
        token_ids=token_ids,
        context_length=context_length
    )

    print()
    print(
        "Dataset size:",
        len(dataset)
    )

    # -----------------------------------------------------
    # Inspect first sample
    # -----------------------------------------------------

    x, y = dataset[0]

    print()
    print(
        "First input:",
        x.tolist()
    )

    print(
        "First target:",
        y.tolist()
    )

    # -----------------------------------------------------
    # Verify shift
    # -----------------------------------------------------

    assert torch.equal(
        y,
        x + 1
    )

    print(
        "Next-token shift verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify shapes
    # -----------------------------------------------------

    assert x.shape == (
        context_length,
    )

    assert y.shape == (
        context_length,
    )

    print(
        "Sample shape verification: PASSED"
    )

    # -----------------------------------------------------
    # Create DataLoader
    # -----------------------------------------------------

    loader = create_dataloader(
        token_ids=token_ids,
        context_length=context_length,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    # -----------------------------------------------------
    # Get first batch
    # -----------------------------------------------------

    batch_x, batch_y = next(
        iter(loader)
    )

    print()
    print(
        "Batch input shape:",
        batch_x.shape
    )

    print(
        "Batch target shape:",
        batch_y.shape
    )

    # -----------------------------------------------------
    # Verify batch shapes
    # -----------------------------------------------------

    assert batch_x.shape == (
        batch_size,
        context_length
    )

    assert batch_y.shape == (
        batch_size,
        context_length
    )

    print(
        "Batch shape verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify target shift
    # -----------------------------------------------------

    assert torch.equal(
        batch_y,
        batch_x + 1
    )

    print(
        "Batch shift verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify dtype
    # -----------------------------------------------------

    assert batch_x.dtype == torch.long
    assert batch_y.dtype == torch.long

    print(
        "Token dtype verification: PASSED"
    )

    # -----------------------------------------------------
    # Important:
    # Dataset stays on CPU.
    #
    # The training loop will move each batch to CUDA.
    # -----------------------------------------------------

    print()
    print(
        "Dataset batch device:",
        batch_x.device
    )

    print(
        "Dataset test: PASSED"
    )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print()
    print("=" * 65)
    print("ALL TESTS PASSED.")
    print("=" * 65)