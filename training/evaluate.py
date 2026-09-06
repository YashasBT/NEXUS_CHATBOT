import os
import sys
import math


# =========================================================
# PROJECT ROOT
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
import torch.nn.functional as F

from model.minigpt import MiniGPT
from tokenizer.tokenizer import CharacterTokenizer


# =========================================================
# PATHS
# =========================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "training.txt"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "nexus_best.pt"
)


# =========================================================
# CONFIGURATION
# =========================================================

CONTEXT_LENGTH = 128
BATCH_SIZE = 8
TRAIN_SPLIT = 0.8


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# LOAD TEXT
# =========================================================

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = CharacterTokenizer(
    text
)

vocab_size = tokenizer.vocab_size


# =========================================================
# ENCODE TEXT
# =========================================================

all_tokens = tokenizer.encode(
    text
)

tokens = torch.tensor(
    all_tokens,
    dtype=torch.long
)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

split_index = int(
    len(tokens) * TRAIN_SPLIT
)

train_tokens = tokens[
    :split_index
]

validation_tokens = tokens[
    split_index:
]


# =========================================================
# CREATE SEQUENCES
# =========================================================

def create_sequences(
    token_data,
    context_length
):

    inputs = []
    targets = []

    for i in range(
        len(token_data) - context_length
    ):

        x = token_data[
            i:i + context_length
        ]

        y = token_data[
            i + 1:i + context_length + 1
        ]

        inputs.append(x)
        targets.append(y)

    if len(inputs) == 0:

        raise ValueError(
            "Not enough tokens to create sequences."
        )

    return (
        torch.stack(inputs),
        torch.stack(targets)
    )


# =========================================================
# BUILD DATASETS
# =========================================================

train_inputs, train_targets = create_sequences(
    train_tokens,
    CONTEXT_LENGTH
)

validation_inputs, validation_targets = create_sequences(
    validation_tokens,
    CONTEXT_LENGTH
)


# =========================================================
# BATCH EVALUATION
# =========================================================

@torch.no_grad()
def evaluate_split(
    model,
    inputs,
    targets,
    batch_size
):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for start in range(
        0,
        len(inputs),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(inputs)
        )

        x = inputs[
            start:end
        ].to(DEVICE)

        y = targets[
            start:end
        ].to(DEVICE)

        # ---------------------------------------------
        # Forward pass
        # ---------------------------------------------

        logits = model(
            x
        )

        # ---------------------------------------------
        # Flatten
        # ---------------------------------------------

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                vocab_size
            ),
            y.reshape(
                -1
            )
        )

        total_loss += loss.item()
        total_batches += 1

    average_loss = (
        total_loss / total_batches
    )

    return average_loss


# =========================================================
# LOAD CHECKPOINT
# =========================================================

if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)


# =========================================================
# CREATE MODEL
# =========================================================

model = MiniGPT(
    vocab_size=vocab_size,
    embedding_dim=checkpoint["embedding_dim"],
    context_length=checkpoint["context_length"],
    num_heads=checkpoint["num_heads"],
    num_blocks=checkpoint["num_blocks"],
    dropout=checkpoint["dropout"],
    expansion_factor=checkpoint["expansion_factor"]
)


# =========================================================
# LOAD PARAMETERS
# =========================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

model.eval()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NEXUS MODEL EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Device information
    # -----------------------------------------------------

    print()
    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

    # -----------------------------------------------------
    # Dataset information
    # -----------------------------------------------------

    print()
    print(
        "Total tokens:",
        len(tokens)
    )

    print(
        "Training tokens:",
        len(train_tokens)
    )

    print(
        "Validation tokens:",
        len(validation_tokens)
    )

    print(
        "Training samples:",
        len(train_inputs)
    )

    print(
        "Validation samples:",
        len(validation_inputs)
    )

    # -----------------------------------------------------
    # Model information
    # -----------------------------------------------------

    model_device = next(
        model.parameters()
    ).device

    print()
    print(
        "Model device:",
        model_device
    )

    assert model_device == DEVICE

    print(
        "Model device verification: PASSED"
    )

    print()
    print(
        "Vocabulary size:",
        vocab_size
    )

    print(
        "Context length:",
        CONTEXT_LENGTH
    )

    # -----------------------------------------------------
    # Evaluate training set
    # -----------------------------------------------------

    print()
    print(
        "Evaluating training set..."
    )

    train_loss = evaluate_split(
        model,
        train_inputs,
        train_targets,
        BATCH_SIZE
    )

    # -----------------------------------------------------
    # Evaluate validation set
    # -----------------------------------------------------

    print(
        "Evaluating validation set..."
    )

    validation_loss = evaluate_split(
        model,
        validation_inputs,
        validation_targets,
        BATCH_SIZE
    )

    # -----------------------------------------------------
    # Perplexity
    # -----------------------------------------------------

    train_perplexity = math.exp(
        train_loss
    )

    validation_perplexity = math.exp(
        validation_loss
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Training loss:       {train_loss:.4f}"
    )

    print(
        f"Validation loss:     {validation_loss:.4f}"
    )

    print()
    print(
        f"Training perplexity: {train_perplexity:.4f}"
    )

    print(
        f"Validation perplexity: "
        f"{validation_perplexity:.4f}"
    )

    # -----------------------------------------------------
    # Checkpoint information
    # -----------------------------------------------------

    print()
    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Checkpoint exists:",
        os.path.exists(
            CHECKPOINT_PATH
        )
    )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    assert math.isfinite(
        train_loss
    )

    assert math.isfinite(
        validation_loss
    )

    assert math.isfinite(
        train_perplexity
    )

    assert math.isfinite(
        validation_perplexity
    )

    print()
    print(
        "Loss verification: PASSED"
    )

    print(
        "Perplexity verification: PASSED"
    )

    if DEVICE.type == "cuda":

        print(
            "CUDA evaluation verification: PASSED"
        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)