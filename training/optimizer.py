import os
import sys

# =========================================================
# ADD NEXUS PROJECT ROOT TO PYTHON PATH
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# IMPORTS
# =========================================================

import torch
import torch.nn as nn

from config import TrainingConfig
from model.minigpt import MiniGPT


# =========================================================
# CREATE OPTIMIZER
# =========================================================

def create_optimizer(model, config):
    """
    Create AdamW optimizer for NEXUS.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    return optimizer


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NEXUS ADAMW OPTIMIZER TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    config = TrainingConfig()

    # Your actual tokenizer vocabulary from Lesson 16
    vocab_size = 43

    config.vocab_size = vocab_size

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = config.device

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
    # Create NEXUS
    # -----------------------------------------------------

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        context_length=config.context_length,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        expansion_factor=config.expansion_factor
    ).to(device)

    # -----------------------------------------------------
    # Verify model device
    # -----------------------------------------------------

    model_device = next(
        model.parameters()
    ).device

    print()
    print(
        "Model device:",
        model_device
    )

    assert model_device.type == device.type

    print(
        "Model device verification: PASSED"
    )

    # -----------------------------------------------------
    # Create AdamW
    # -----------------------------------------------------

    optimizer = create_optimizer(
        model=model,
        config=config
    )

    print()
    print(
        "Optimizer:",
        type(optimizer).__name__
    )

    print(
        "Learning rate:",
        config.learning_rate
    )

    print(
        "Weight decay:",
        config.weight_decay
    )

    # -----------------------------------------------------
    # Create training batch
    # -----------------------------------------------------

    batch_size = config.batch_size
    sequence_length = config.context_length

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length
        ),
        dtype=torch.long,
        device=device
    )

    y = torch.randint(
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
        "Input shape:",
        x.shape
    )

    print(
        "Target shape:",
        y.shape
    )

    print(
        "Input device:",
        x.device
    )

    print(
        "Target device:",
        y.device
    )

    assert x.device.type == device.type
    assert y.device.type == device.type

    print(
        "Batch device verification: PASSED"
    )

    # -----------------------------------------------------
    # Save parameter BEFORE update
    # -----------------------------------------------------

    parameter_before = (
        model.token_embedding
        .embedding
        .weight
        .detach()
        .clone()
    )

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    logits = model(x)

    print()
    print(
        "Logits shape:",
        logits.shape
    )

    assert logits.shape == (
        batch_size,
        sequence_length,
        vocab_size
    )

    print(
        "Logits shape verification: PASSED"
    )

    # -----------------------------------------------------
    # Cross entropy
    # -----------------------------------------------------

    loss_function = nn.CrossEntropyLoss()

    loss = loss_function(
        logits.reshape(
            -1,
            vocab_size
        ),
        y.reshape(-1)
    )

    print()
    print(
        "Loss before update:",
        loss.item()
    )

    # -----------------------------------------------------
    # Clear gradients
    # -----------------------------------------------------

    optimizer.zero_grad()

    # -----------------------------------------------------
    # Backpropagation
    # -----------------------------------------------------

    loss.backward()

    print(
        "Backward pass: PASSED"
    )

    # -----------------------------------------------------
    # Gradient clipping
    # -----------------------------------------------------

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=config.gradient_clip
    )

    print(
        "Gradient norm:",
        gradient_norm.item()
    )

    print(
        "Gradient clipping: PASSED"
    )

    # -----------------------------------------------------
    # AdamW UPDATE
    # -----------------------------------------------------

    optimizer.step()

    print(
        "AdamW optimizer step: PASSED"
    )

    # -----------------------------------------------------
    # Save parameter AFTER update
    # -----------------------------------------------------

    parameter_after = (
        model.token_embedding
        .embedding
        .weight
        .detach()
        .clone()
    )

    # -----------------------------------------------------
    # Verify weights actually changed
    # -----------------------------------------------------

    parameter_difference = (
        parameter_after - parameter_before
    ).abs().sum().item()

    print()
    print(
        "Parameter change:",
        parameter_difference
    )

    assert parameter_difference > 0

    print(
        "Parameter update verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify optimizer state
    # -----------------------------------------------------

    optimizer_state_count = len(
        optimizer.state
    )

    print(
        "Optimizer state entries:",
        optimizer_state_count
    )

    assert optimizer_state_count > 0

    print(
        "Optimizer state verification: PASSED"
    )

    # -----------------------------------------------------
    # Verify CUDA
    # -----------------------------------------------------

    if torch.cuda.is_available():

        assert device.type == "cuda"

        print(
            "CUDA verification: PASSED"
        )

        print(
            "Training computation device: cuda:0"
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
        "NEXUS parameters:",
        f"{total_parameters:,}"
    )

    # -----------------------------------------------------
    # FINAL
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("ALL TESTS PASSED.")
    print("=" * 70)