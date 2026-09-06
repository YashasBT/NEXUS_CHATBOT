import os
import sys


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
import torch.nn as nn

from torch.utils.data import DataLoader

from model.minigpt import MiniGPT
from tokenizer.tokenizer import CharacterTokenizer
from dataset import NextTokenDataset
from config import TrainingConfig


# =========================================================
# CONFIGURATION
# =========================================================

config = TrainingConfig()

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "training.txt"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "nexus_latest.pt"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "nexus_best.pt"
)

TRAIN_SPLIT = config.train_split

CONTEXT_LENGTH = config.context_length

BATCH_SIZE = config.batch_size

DEVICE = config.device


# =========================================================
# RESUME OPTION
# =========================================================

# False = start a completely new training run
# True  = continue from nexus_latest.pt

RESUME = False


# =========================================================
# CREATE CHECKPOINT DIRECTORY
# =========================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD TRAINING TEXT
# =========================================================

def load_training_text():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Training file not found:\n{DATA_PATH}"
        )

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    if not text.strip():

        raise ValueError(
            "training.txt is empty."
        )

    return text


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(text):

    # -----------------------------------------------------
    # Create tokenizer
    # -----------------------------------------------------

    tokenizer = CharacterTokenizer(text)

    # -----------------------------------------------------
    # Encode text
    # -----------------------------------------------------

    token_ids = tokenizer.encode(text)

    token_ids = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    # -----------------------------------------------------
    # Vocabulary size
    # -----------------------------------------------------

    vocab_size = tokenizer.vocab_size

    # -----------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------

    split_index = int(
        len(token_ids) * TRAIN_SPLIT
    )

    train_tokens = token_ids[
        :split_index
    ]

    validation_tokens = token_ids[
        split_index:
    ]

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = NextTokenDataset(
        token_ids=train_tokens,
        context_length=CONTEXT_LENGTH
    )

    validation_dataset = NextTokenDataset(
        token_ids=validation_tokens,
        context_length=CONTEXT_LENGTH
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True
    )

    return (
        tokenizer,
        vocab_size,
        train_loader,
        validation_loader
    )


# =========================================================
# CREATE MODEL
# =========================================================

def create_model(vocab_size):

    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        context_length=config.context_length,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        expansion_factor=config.expansion_factor
    )

    model = model.to(DEVICE)

    return model


# =========================================================
# CREATE OPTIMIZER
# =========================================================

def create_optimizer(model):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    return optimizer


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_function
):

    model.train()

    total_loss = 0.0

    number_of_batches = 0

    for batch_index, (x, y) in enumerate(
        train_loader
    ):

        # -------------------------------------------------
        # MOVE DATA TO GPU
        # -------------------------------------------------

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        # -------------------------------------------------
        # CLEAR GRADIENTS
        # -------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # -------------------------------------------------
        # FORWARD
        # -------------------------------------------------

        logits = model(x)

        # -------------------------------------------------
        # LOSS
        # -------------------------------------------------

        loss = loss_function(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            y.reshape(-1)
        )

        # -------------------------------------------------
        # BACKWARD
        # -------------------------------------------------

        loss.backward()

        # -------------------------------------------------
        # GRADIENT CLIPPING
        # -------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.gradient_clip
        )

        # -------------------------------------------------
        # OPTIMIZER UPDATE
        # -------------------------------------------------

        optimizer.step()

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        total_loss += loss.item()

        number_of_batches += 1

        # -------------------------------------------------
        # PROGRESS
        # -------------------------------------------------

        if (
            batch_index == 0
            or (batch_index + 1) % 10 == 0
        ):

            print(
                f"  Batch "
                f"{batch_index + 1:4d}/"
                f"{len(train_loader):4d}"
                f" | Loss: {loss.item():.4f}"
            )

    return (
        total_loss /
        number_of_batches
    )


# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def validate(
    model,
    validation_loader,
    loss_function
):

    model.eval()

    total_loss = 0.0

    number_of_batches = 0

    for x, y in validation_loader:

        # -------------------------------------------------
        # MOVE TO GPU
        # -------------------------------------------------

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        # -------------------------------------------------
        # FORWARD
        # -------------------------------------------------

        logits = model(x)

        # -------------------------------------------------
        # LOSS
        # -------------------------------------------------

        loss = loss_function(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            y.reshape(-1)
        )

        total_loss += loss.item()

        number_of_batches += 1

    if number_of_batches == 0:

        return float("inf")

    return (
        total_loss /
        number_of_batches
    )


# =========================================================
# SAVE CHECKPOINT
# =========================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    vocab_size
):

    checkpoint = {

        # Model parameters
        "model_state_dict":
            model.state_dict(),

        # Optimizer parameters
        "optimizer_state_dict":
            optimizer.state_dict(),

        # Training progress
        "epoch":
            epoch,

        # Loss information
        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        # Model/data information
        "vocab_size":
            vocab_size,

        "context_length":
            config.context_length,

        "embedding_dim":
            config.embedding_dim,

        "num_heads":
            config.num_heads,

        "num_blocks":
            config.num_blocks,

        "dropout":
            config.dropout,

        "expansion_factor":
            config.expansion_factor
    }

    torch.save(
        checkpoint,
        path
    )


# =========================================================
# LOAD CHECKPOINT
# =========================================================

def load_checkpoint(
    path,
    model,
    optimizer
):

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = (
        checkpoint["epoch"] + 1
    )

    train_loss = checkpoint.get(
        "train_loss",
        None
    )

    validation_loss = checkpoint.get(
        "validation_loss",
        None
    )

    return (
        start_epoch,
        train_loss,
        validation_loss
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NEXUS GPU TRAINING + CHECKPOINTING")
    print("=" * 70)

    # -----------------------------------------------------
    # DEVICE
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
    # LOAD DATA
    # -----------------------------------------------------

    print()
    print(
        "Loading training data..."
    )

    text = load_training_text()

    print(
        "Character count:",
        len(text)
    )

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    print()
    print(
        "Preparing tokenizer and datasets..."
    )

    (
        tokenizer,
        vocab_size,
        train_loader,
        validation_loader
    ) = prepare_data(text)

    print(
        "Vocabulary size:",
        vocab_size
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(validation_loader)
    )

    # -----------------------------------------------------
    # CREATE MODEL
    # -----------------------------------------------------

    print()
    print(
        "Creating NEXUS..."
    )

    model = create_model(
        vocab_size
    )

    model_device = next(
        model.parameters()
    ).device

    print(
        "Model device:",
        model_device
    )

    assert model_device == DEVICE

    print(
        "Model device verification: PASSED"
    )

    # -----------------------------------------------------
    # CREATE OPTIMIZER
    # -----------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    loss_function = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # PARAMETER COUNT
    # -----------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "NEXUS parameters:",
        f"{total_parameters:,}"
    )

    # -----------------------------------------------------
    # TRAINING STATE
    # -----------------------------------------------------

    start_epoch = 1

    best_validation_loss = float(
        "inf"
    )

    # -----------------------------------------------------
    # RESUME
    # -----------------------------------------------------

    if RESUME:

        if not os.path.exists(
            LATEST_CHECKPOINT
        ):

            raise FileNotFoundError(
                "RESUME=True but checkpoint "
                "does not exist:\n"
                f"{LATEST_CHECKPOINT}"
            )

        print()
        print(
            "Loading checkpoint..."
        )

        (
            start_epoch,
            previous_train_loss,
            previous_validation_loss
        ) = load_checkpoint(
            LATEST_CHECKPOINT,
            model,
            optimizer
        )

        print(
            "Checkpoint loaded: PASSED"
        )

        print(
            "Resuming from epoch:",
            start_epoch
        )

        print(
            "Previous training loss:",
            previous_train_loss
        )

        print(
            "Previous validation loss:",
            previous_validation_loss
        )

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        start_epoch,
        config.epochs + 1
    ):

        print()
        print(
            f"Epoch "
            f"{epoch}/"
            f"{config.epochs}"
        )

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function
        )

        # -------------------------------------------------
        # VALIDATE
        # -------------------------------------------------

        validation_loss = validate(
            model=model,
            validation_loader=validation_loader,
            loss_function=loss_function
        )

        # -------------------------------------------------
        # PRINT RESULTS
        # -------------------------------------------------

        print()
        print(
            f"Epoch {epoch} complete"
        )

        print(
            f"Training loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        # -------------------------------------------------
        # SAVE LATEST
        # -------------------------------------------------

        save_checkpoint(
            path=LATEST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            vocab_size=vocab_size
        )

        print()
        print(
            "Latest checkpoint saved:"
        )

        print(
            LATEST_CHECKPOINT
        )

        # -------------------------------------------------
        # SAVE BEST
        # -------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                vocab_size=vocab_size
            )

            print(
                "New best model: SAVED"
            )

            print(
                BEST_CHECKPOINT
            )

    # -----------------------------------------------------
    # CUDA SYNCHRONIZATION
    # -----------------------------------------------------

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    # -----------------------------------------------------
    # FINAL
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    if "train_loss" in locals():

        print(
            "Final training loss:",
            f"{train_loss:.4f}"
        )

        print(
            "Final validation loss:",
            f"{validation_loss:.4f}"
        )

    print(
        "Training device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print(
        "Latest checkpoint exists:",
        os.path.exists(
            LATEST_CHECKPOINT
        )
    )

    print(
        "Best checkpoint exists:",
        os.path.exists(
            BEST_CHECKPOINT
        )
    )

    print()
    print("=" * 70)
    print("CHECKPOINTING TEST COMPLETE.")
    print("=" * 70)