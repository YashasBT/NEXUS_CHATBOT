import os
import sys
import argparse


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
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# LOAD TRAINING TEXT
# =========================================================

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


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = CharacterTokenizer(
    text
)

vocab_size = tokenizer.vocab_size


# =========================================================
# CHECKPOINT VALIDATION
# =========================================================

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"NEXUS checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


# =========================================================
# LOAD CHECKPOINT
# =========================================================

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
# LOAD TRAINED WEIGHTS
# =========================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# =========================================================
# MOVE MODEL TO GPU
# =========================================================

model = model.to(
    DEVICE
)

model.eval()


# =========================================================
# TOP-K FILTERING
# =========================================================

def apply_top_k(
    logits,
    top_k
):
    """
    Keep only the top-k highest-probability tokens.
    """

    if top_k is None:
        return logits

    if top_k <= 0:
        return logits

    top_k = min(
        top_k,
        logits.size(-1)
    )

    values, _ = torch.topk(
        logits,
        top_k,
        dim=-1
    )

    minimum_value = values[
        :, -1
    ].unsqueeze(-1)

    logits = torch.where(
        logits < minimum_value,
        torch.full_like(
            logits,
            float("-inf")
        ),
        logits
    )

    return logits


# =========================================================
# TOP-P / NUCLEUS SAMPLING
# =========================================================

def apply_top_p(
    logits,
    top_p
):
    """
    Keep the smallest group of tokens whose
    cumulative probability reaches top_p.
    """

    if top_p is None:
        return logits

    if top_p >= 1.0:
        return logits

    if top_p <= 0.0:
        raise ValueError(
            "top_p must be greater than 0."
        )

    # -----------------------------------------------------
    # Sort logits from highest to lowest
    # -----------------------------------------------------

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1
    )

    # -----------------------------------------------------
    # Convert to probabilities
    # -----------------------------------------------------

    sorted_probabilities = torch.softmax(
        sorted_logits,
        dim=-1
    )

    # -----------------------------------------------------
    # Cumulative probability
    # -----------------------------------------------------

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1
    )

    # -----------------------------------------------------
    # Remove tokens beyond top-p
    # -----------------------------------------------------

    sorted_indices_to_remove = (
        cumulative_probabilities > top_p
    )

    # Always keep the highest-probability token
    sorted_indices_to_remove[
        :, 0
    ] = False

    # -----------------------------------------------------
    # Convert sorted mask back to vocabulary order
    # -----------------------------------------------------

    indices_to_remove = torch.zeros_like(
        sorted_indices_to_remove
    )

    indices_to_remove.scatter_(
        1,
        sorted_indices,
        sorted_indices_to_remove
    )

    # -----------------------------------------------------
    # Mask removed tokens
    # -----------------------------------------------------

    logits = logits.masked_fill(
        indices_to_remove,
        float("-inf")
    )

    return logits


# =========================================================
# REPETITION PENALTY
# =========================================================

def apply_repetition_penalty(
    logits,
    tokens,
    repetition_penalty
):
    """
    Penalize tokens that have already appeared.
    """

    if repetition_penalty == 1.0:
        return logits

    if repetition_penalty <= 0.0:
        raise ValueError(
            "repetition_penalty must be greater than 0."
        )

    # -----------------------------------------------------
    # Find unique tokens that already occurred
    # -----------------------------------------------------

    previous_tokens = torch.unique(
        tokens
    )

    # -----------------------------------------------------
    # Apply penalty
    # -----------------------------------------------------

    for token_id in previous_tokens:

        token_id = token_id.item()

        if logits[
            0,
            token_id
        ] < 0:

            logits[
                0,
                token_id
            ] *= repetition_penalty

        else:

            logits[
                0,
                token_id
            ] /= repetition_penalty

    return logits


# =========================================================
# TEXT GENERATION
# =========================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=10,
    top_p=0.9,
    repetition_penalty=1.1
):
    """
    Generate text autoregressively from a prompt.
    """

    # =====================================================
    # VALIDATION
    # =====================================================

    if not prompt:
        raise ValueError(
            "Prompt cannot be empty."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than 0."
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than 0."
        )

    if top_p <= 0 or top_p > 1:
        raise ValueError(
            "top_p must be in the range (0, 1]."
        )

    if repetition_penalty <= 0:
        raise ValueError(
            "repetition_penalty must be greater than 0."
        )

    # =====================================================
    # ENCODE PROMPT
    # =====================================================

    token_ids = tokenizer.encode(
        prompt
    )

    if len(token_ids) == 0:
        raise ValueError(
            "Prompt produced no tokens."
        )

    # =====================================================
    # CREATE TOKEN TENSOR
    # =====================================================

    tokens = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    # Shape:
    #
    # [1, prompt_length]

    # =====================================================
    # AUTOREGRESSIVE GENERATION LOOP
    # =====================================================

    for _ in range(
        max_new_tokens
    ):

        # -------------------------------------------------
        # Context window
        # -------------------------------------------------

        input_tokens = tokens[
            :,
            -model.context_length:
        ]

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        logits = model(
            input_tokens
        )

        # -------------------------------------------------
        # Take logits from final position
        # -------------------------------------------------

        logits = logits[
            :,
            -1,
            :
        ]

        # Shape:
        #
        # [1, vocab_size]

        # -------------------------------------------------
        # Repetition penalty
        # -------------------------------------------------

        logits = apply_repetition_penalty(
            logits,
            tokens[0],
            repetition_penalty
        )

        # -------------------------------------------------
        # Temperature
        # -------------------------------------------------

        logits = logits / temperature

        # -------------------------------------------------
        # Top-k
        # -------------------------------------------------

        logits = apply_top_k(
            logits,
            top_k
        )

        # -------------------------------------------------
        # Top-p
        # -------------------------------------------------

        logits = apply_top_p(
            logits,
            top_p
        )

        # -------------------------------------------------
        # Convert logits to probabilities
        # -------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if torch.isnan(
            probabilities
        ).any():

            raise RuntimeError(
                "NaN detected in sampling probabilities."
            )

        # -------------------------------------------------
        # Sample next token
        # -------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Shape:
        #
        # [1, 1]

        # -------------------------------------------------
        # Append next token
        # -------------------------------------------------

        tokens = torch.cat(
            [
                tokens,
                next_token
            ],
            dim=1
        )

    # =====================================================
    # DECODE
    # =====================================================

    generated_token_ids = tokens[
        0
    ].tolist()

    generated_text = tokenizer.decode(
        generated_token_ids
    )

    return generated_text


# =========================================================
# COMMAND-LINE ARGUMENTS
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="NEXUS text generation engine"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Lower = more deterministic."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-probability tokens to consider."
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability."
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to previously generated tokens."
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate."
    )

    return parser.parse_args()


# =========================================================
# DISPLAY SETTINGS
# =========================================================

def print_generation_settings(
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty
):

    print()
    print(
        "Generation settings:"
    )

    print(
        f"  Max new tokens:      {max_new_tokens}"
    )

    print(
        f"  Temperature:         {temperature}"
    )

    print(
        f"  Top-k:               {top_k}"
    )

    print(
        f"  Top-p:               {top_p}"
    )

    print(
        f"  Repetition penalty:  {repetition_penalty}"
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NEXUS ADVANCED TEXT GENERATION")
    print("=" * 70)

    # =====================================================
    # DEVICE INFORMATION
    # =====================================================

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

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

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

    print(
        "Vocabulary size:",
        vocab_size
    )

    print(
        "Context length:",
        model.context_length
    )

    model_device = next(
        model.parameters()
    ).device

    print(
        "Model device:",
        model_device
    )

    # =====================================================
    # DEVICE VERIFICATION
    # =====================================================

    assert model_device == DEVICE

    print(
        "Model device verification: PASSED"
    )

    print(
        "Model loaded successfully."
    )

    # =====================================================
    # ARGUMENTS
    # =====================================================

    args = parse_arguments()

    max_new_tokens = args.max_new_tokens

    temperature = args.temperature

    top_k = args.top_k

    top_p = args.top_p

    repetition_penalty = (
        args.repetition_penalty
    )

    # =====================================================
    # DISPLAY SETTINGS
    # =====================================================

    print_generation_settings(
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty
    )

    # =====================================================
    # INTERACTIVE MODE
    # =====================================================

    print()

    print("=" * 70)
    print("INTERACTIVE GENERATION")
    print("=" * 70)

    print()

    print(
        "Type a prompt and press Enter."
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    while True:

        try:

            prompt = input(
                "Prompt: "
            )

        except KeyboardInterrupt:

            print()
            print(
                "Generation stopped."
            )
            break

        except EOFError:

            print()
            break

        # -------------------------------------------------
        # EXIT
        # -------------------------------------------------

        if prompt.strip().lower() == "exit":

            print(
                "Exiting NEXUS."
            )

            break

        # -------------------------------------------------
        # EMPTY PROMPT
        # -------------------------------------------------

        if not prompt.strip():

            print(
                "Please enter a prompt."
            )

            print()

            continue

        # -------------------------------------------------
        # GENERATE
        # -------------------------------------------------

        print()

        print(
            "Generating..."
        )

        try:

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        except Exception as error:

            print()

            print(
                "Generation error:"
            )

            print(
                error
            )

            print()

            continue

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        print()

        print(
            "NEXUS:"
        )

        print(
            generated_text
        )

        print()