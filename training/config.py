import torch


class TrainingConfig:
    """
    Configuration for NEXUS training.
    """

    # =====================================================
    # Data
    # =====================================================

    data_path = "data/raw/training.txt"

    context_length = 128

    batch_size = 8

    train_split = 0.80

    # =====================================================
    # Model
    # =====================================================

    embedding_dim = 128

    num_heads = 4

    num_blocks = 4

    dropout = 0.1

    expansion_factor = 4

    # =====================================================
    # Vocabulary
    #
    # This will be replaced with the actual tokenizer
    # vocabulary size during training.
    # =====================================================

    vocab_size = None

    # =====================================================
    # Optimizer
    # =====================================================

    learning_rate = 3e-4

    weight_decay = 0.01

    # =====================================================
    # Training
    # =====================================================

    epochs = 2

    gradient_clip = 1.0

    # =====================================================
    # Device
    # =====================================================

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


def print_config(config):
    """
    Display training configuration.
    """

    print("=" * 65)
    print("NEXUS TRAINING CONFIGURATION")
    print("=" * 65)

    print()

    print("Device:", config.device)

    if config.device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

    print()

    print("----- DATA -----")

    print(
        "Context length:",
        config.context_length
    )

    print(
        "Batch size:",
        config.batch_size
    )

    print(
        "Train split:",
        config.train_split
    )

    print()

    print("----- MODEL -----")

    print(
        "Embedding dimension:",
        config.embedding_dim
    )

    print(
        "Attention heads:",
        config.num_heads
    )

    print(
        "Transformer blocks:",
        config.num_blocks
    )

    print(
        "Dropout:",
        config.dropout
    )

    print(
        "Expansion factor:",
        config.expansion_factor
    )

    print(
        "Vocabulary size:",
        config.vocab_size
    )

    print()

    print("----- OPTIMIZER -----")

    print(
        "Learning rate:",
        config.learning_rate
    )

    print(
        "Weight decay:",
        config.weight_decay
    )

    print()

    print("----- TRAINING -----")

    print(
        "Epochs:",
        config.epochs
    )

    print(
        "Gradient clipping:",
        config.gradient_clip
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    config = TrainingConfig()

    print_config(config)

    print()

    # -----------------------------------------------------
    # Device verification
    # -----------------------------------------------------

    assert config.device.type in (
        "cuda",
        "cpu"
    )

    print(
        "Device configuration: PASSED"
    )

    # -----------------------------------------------------
    # CUDA verification
    # -----------------------------------------------------

    if torch.cuda.is_available():

        assert config.device.type == "cuda"

        print(
            "CUDA configuration: PASSED"
        )

    # -----------------------------------------------------
    # Model configuration verification
    # -----------------------------------------------------

    assert config.embedding_dim == 128

    assert config.num_heads == 4

    assert config.num_blocks == 4

    assert config.context_length == 128

    print(
        "Model configuration: PASSED"
    )

    # -----------------------------------------------------
    # Optimizer configuration verification
    # -----------------------------------------------------

    assert config.learning_rate > 0

    assert config.weight_decay >= 0

    print(
        "Optimizer configuration: PASSED"
    )

    print()

    print("=" * 65)
    print("CONFIGURATION TEST PASSED.")
    print("=" * 65)