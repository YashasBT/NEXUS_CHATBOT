import os
import sys

import torch
from torch.utils.data import DataLoader


# =========================================================
# NEXUS PROJECT PATH
# =========================================================

# data_preparation.py is inside:
#
# D:\NEXUS\training\data_preparation.py
#
# Therefore the NEXUS root is one directory above "training".

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# IMPORT NEXT-TOKEN DATASET
# =========================================================

from dataset import NextTokenDataset


# =========================================================
# IMPORT TOKENIZER
# =========================================================

try:
    from tokenizer.tokenizer import CharacterTokenizer
except ImportError as error:
    raise ImportError(
        "\nCould not import CharacterTokenizer.\n\n"
        "Expected file:\n"
        f"{os.path.join(PROJECT_ROOT, 'tokenizer', 'tokenizer.py')}\n\n"
        "Make sure tokenizer.py contains a class named:\n"
        "CharacterTokenizer\n"
    ) from error


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "training.txt"
)

CONTEXT_LENGTH = 128

BATCH_SIZE = 8

TRAIN_SPLIT = 0.80


# =========================================================
# LOAD TEXT
# =========================================================

def load_text(file_path):
    """
    Load training text from disk.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Training file not found:\n{file_path}"
        )

    with open(
        file_path,
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
# TOKENIZER
# =========================================================

def build_tokenizer(text):
    """
    Create the CharacterTokenizer.

    The NEXUS CharacterTokenizer expects the
    training text directly in its constructor.
    """

    tokenizer = CharacterTokenizer(text)

    return tokenizer


# =========================================================
# GET VOCABULARY SIZE
# =========================================================

def get_vocab_size(tokenizer):
    """
    Obtain vocabulary size from the tokenizer.
    """

    if hasattr(tokenizer, "vocab_size"):

        vocab_size = tokenizer.vocab_size

        # Handle property or method.
        if callable(vocab_size):
            vocab_size = vocab_size()

        return int(vocab_size)

    if hasattr(tokenizer, "stoi"):

        return len(tokenizer.stoi)

    if hasattr(tokenizer, "char_to_id"):

        return len(tokenizer.char_to_id)

    if hasattr(tokenizer, "token_to_id"):

        return len(tokenizer.token_to_id)

    if hasattr(tokenizer, "vocab"):

        return len(tokenizer.vocab)

    raise AttributeError(
        "Could not determine tokenizer vocabulary size."
    )


# =========================================================
# ENCODE TEXT
# =========================================================

def encode_text(tokenizer, text):
    """
    Encode text into token IDs.
    """

    if not hasattr(tokenizer, "encode"):
        raise AttributeError(
            "CharacterTokenizer does not contain encode()."
        )

    token_ids = tokenizer.encode(text)

    if isinstance(token_ids, torch.Tensor):

        token_ids = token_ids.to(
            dtype=torch.long
        )

    else:

        token_ids = torch.tensor(
            token_ids,
            dtype=torch.long
        )

    return token_ids


# =========================================================
# DECODE TOKENS
# =========================================================

def decode_tokens(tokenizer, token_ids):
    """
    Decode token IDs back into text.
    """

    if not hasattr(tokenizer, "decode"):
        return None

    return tokenizer.decode(
        token_ids
    )


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

def split_tokens(
    token_ids,
    train_split=TRAIN_SPLIT
):
    """
    Split token stream chronologically.

    First 90%:
        Training

    Last 10%:
        Validation
    """

    if not 0.0 < train_split < 1.0:
        raise ValueError(
            "train_split must be between 0 and 1."
        )

    split_index = int(
        len(token_ids) * train_split
    )

    if split_index <= CONTEXT_LENGTH:
        raise ValueError(
            "Training data is too small for "
            f"context length {CONTEXT_LENGTH}."
        )

    if (
        len(token_ids) - split_index
        <= CONTEXT_LENGTH
    ):
        raise ValueError(
            "Validation data is too small for "
            f"context length {CONTEXT_LENGTH}."
        )

    train_tokens = token_ids[
        :split_index
    ]

    validation_tokens = token_ids[
        split_index:
    ]

    return (
        train_tokens,
        validation_tokens
    )


# =========================================================
# CREATE DATALOADERS
# =========================================================

def create_loaders(
    train_tokens,
    validation_tokens,
    context_length,
    batch_size
):
    """
    Create training and validation datasets/loaders.
    """

    train_dataset = NextTokenDataset(
        token_ids=train_tokens,
        context_length=context_length
    )

    validation_dataset = NextTokenDataset(
        token_ids=validation_tokens,
        context_length=context_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    return (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NEXUS REAL DATA PREPARATION")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load training text
    # -----------------------------------------------------

    text = load_text(
        DATA_PATH
    )

    print()
    print(
        "Training file:",
        DATA_PATH
    )

    print(
        "Character count:",
        len(text)
    )

    # -----------------------------------------------------
    # 2. Show text preview
    # -----------------------------------------------------

    print()
    print("First 200 characters:")

    print(
        repr(text[:200])
    )

    # -----------------------------------------------------
    # 3. Create tokenizer
    # -----------------------------------------------------

    print()
    print("Building tokenizer...")

    tokenizer = build_tokenizer(
        text
    )

    print(
        "Tokenizer: PASSED"
    )

    # -----------------------------------------------------
    # 4. Get vocabulary size
    # -----------------------------------------------------

    vocab_size = get_vocab_size(
        tokenizer
    )

    print(
        "Vocabulary size:",
        vocab_size
    )

    # -----------------------------------------------------
    # 5. Encode entire training file
    # -----------------------------------------------------

    print()
    print("Encoding training text...")

    token_ids = encode_text(
        tokenizer,
        text
    )

    print(
        "Encoding: PASSED"
    )

    print(
        "Total token count:",
        len(token_ids)
    )

    print(
        "Token tensor shape:",
        token_ids.shape
    )

    print(
        "Token tensor dtype:",
        token_ids.dtype
    )

    # -----------------------------------------------------
    # 6. Verify token tensor
    # -----------------------------------------------------

    assert token_ids.dtype == torch.long

    assert token_ids.dim() == 1

    assert len(token_ids) > CONTEXT_LENGTH

    print(
        "Token tensor verification: PASSED"
    )

    # -----------------------------------------------------
    # 7. Decode preview
    # -----------------------------------------------------

    decoded_preview = decode_tokens(
        tokenizer,
        token_ids[:100].tolist()
    )

    if decoded_preview is not None:

        print()
        print("Decoded preview:")

        print(
            repr(decoded_preview)
        )

        print(
            "Encode/decode verification: PASSED"
        )

    # -----------------------------------------------------
    # 8. Verify token range
    # -----------------------------------------------------

    assert token_ids.min().item() >= 0

    assert (
        token_ids.max().item()
        < vocab_size
    )

    print(
        "Token range verification: PASSED"
    )

    # -----------------------------------------------------
    # 9. Train/validation split
    # -----------------------------------------------------

    (
        train_tokens,
        validation_tokens
    ) = split_tokens(
        token_ids,
        train_split=TRAIN_SPLIT
    )

    print()
    print(
        "Training tokens:",
        len(train_tokens)
    )

    print(
        "Validation tokens:",
        len(validation_tokens)
    )

    # -----------------------------------------------------
    # 10. Create DataLoaders
    # -----------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    ) = create_loaders(
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE
    )

    print()
    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    # -----------------------------------------------------
    # 11. Get training batch
    # -----------------------------------------------------

    train_x, train_y = next(
        iter(train_loader)
    )

    print()
    print(
        "Training input shape:",
        train_x.shape
    )

    print(
        "Training target shape:",
        train_y.shape
    )

    # -----------------------------------------------------
    # 12. Verify batch shape
    # -----------------------------------------------------

    assert train_x.shape == (
        BATCH_SIZE,
        CONTEXT_LENGTH
    )

    assert train_y.shape == (
        BATCH_SIZE,
        CONTEXT_LENGTH
    )

    print(
        "Batch shape verification: PASSED"
    )

    # -----------------------------------------------------
    # 13. Verify next-token relationship
    # -----------------------------------------------------

    assert torch.equal(
        train_x[:, 1:],
        train_y[:, :-1]
    )

    print(
        "Next-token relationship: PASSED"
    )

    # -----------------------------------------------------
    # 14. Verify dtype
    # -----------------------------------------------------

    assert train_x.dtype == torch.long

    assert train_y.dtype == torch.long

    print(
        "Token dtype verification: PASSED"
    )

    # -----------------------------------------------------
    # 15. Verify token ranges
    # -----------------------------------------------------

    assert train_x.min().item() >= 0

    assert train_y.min().item() >= 0

    assert (
        train_x.max().item()
        < vocab_size
    )

    assert (
        train_y.max().item()
        < vocab_size
    )

    print(
        "Batch token range verification: PASSED"
    )

    # -----------------------------------------------------
    # 16. Dataset device
    # -----------------------------------------------------

    print()
    print(
        "DataLoader batch device:",
        train_x.device
    )

    # Dataset intentionally stays on CPU.
    assert train_x.device.type == "cpu"

    print(
        "CPU dataset verification: PASSED"
    )

    # -----------------------------------------------------
    # 17. CUDA availability
    # -----------------------------------------------------

    print()
    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

    # -----------------------------------------------------
    # FINAL
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("DATA PREPARATION TEST PASSED.")
    print("=" * 70)