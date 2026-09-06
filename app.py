# ============================================================
# NEXUS - Neural Intelligence Terminal
# Streamlit Application
# ============================================================

import inspect
import math
import time
from pathlib import Path

import streamlit as st
import torch


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NEXUS // Neural Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "nexus_best.pt"
LATEST_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "nexus_latest.pt"

TRAINING_DATA_PATH = BASE_DIR / "data" / "raw" / "training.txt"


# ============================================================
# NEXUS CONFIGURATION
# ============================================================

VOCAB_SIZE_DEFAULT = 86
CONTEXT_LENGTH = 128
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


DEVICE_NAME = str(DEVICE)

if DEVICE.type == "cuda":
    GPU_NAME = torch.cuda.get_device_name(DEVICE)
    CUDA_VERSION = torch.version.cuda
else:
    GPU_NAME = "CPU"
    CUDA_VERSION = "N/A"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
}

.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(30, 70, 100, 0.12),
            transparent 35%
        ),
        #080b10;
    color: #e6eaf2;
}

section[data-testid="stSidebar"] {
    background: #090d12;
    border-right: 1px solid #202833;
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

.nexus-brand {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 8px;
    color: #f2f5fa;
    margin-bottom: 4px;
}

.nexus-subtitle {
    color: #718095;
    font-size: 10px;
    letter-spacing: 3px;
    margin-bottom: 25px;
}

.section-label {
    color: #657285;
    font-size: 10px;
    letter-spacing: 3px;
    margin-top: 25px;
    margin-bottom: 10px;
}

.status-online {
    color: #45ffd2;
    font-weight: 700;
    letter-spacing: 2px;
}

.status-offline {
    color: #ff5f65;
    font-weight: 700;
    letter-spacing: 2px;
}

.status-dot {
    font-size: 16px;
}

.terminal-header {
    border-bottom: 1px solid #222b36;
    padding-bottom: 18px;
    margin-bottom: 24px;
}

.terminal-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 5px;
    color: #f1f4f8;
}

.terminal-subtitle {
    margin-top: 7px;
    color: #687587;
    font-size: 11px;
    letter-spacing: 2px;
}

.terminal-meta {
    color: #7b8797;
    font-size: 11px;
    line-height: 1.8;
    text-align: right;
}

.meta-accent {
    color: #45ffd2;
}

.info-panel {
    background: #11161d;
    border: 1px solid #27313d;
    padding: 18px;
    margin-bottom: 18px;
}

.info-title {
    color: #687587;
    font-size: 10px;
    letter-spacing: 3px;
    margin-bottom: 10px;
}

.info-value {
    color: #e8edf4;
    font-size: 14px;
}

.metric-box {
    background: #10151c;
    border: 1px solid #27313d;
    padding: 18px;
    min-height: 110px;
}

.metric-title {
    color: #687587;
    font-size: 9px;
    letter-spacing: 3px;
}

.metric-value {
    color: #f1f4f8;
    font-size: 27px;
    margin-top: 8px;
}

.metric-accent {
    color: #45ffd2;
}

.chat-user {
    border-left: 2px solid #45ffd2;
    background: #0e141b;
    padding: 15px 18px;
    margin: 12px 0;
}

.chat-nexus {
    border-left: 2px solid #6488ff;
    background: #10151d;
    padding: 15px 18px;
    margin: 12px 0 24px 0;
}

.chat-label {
    color: #657285;
    font-size: 9px;
    letter-spacing: 3px;
    margin-bottom: 8px;
}

.chat-text {
    color: #e5eaf1;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
}

.empty-terminal {
    text-align: center;
    padding: 70px 20px;
    border: 1px solid #202a35;
    background: #0d1218;
}

.empty-title {
    color: #cdd5df;
    font-size: 14px;
    letter-spacing: 3px;
}

.empty-subtitle {
    color: #687587;
    font-size: 11px;
    margin-top: 12px;
}

.error-panel {
    border: 1px solid #6d2d34;
    background: #281519;
    padding: 18px;
    color: #ff747b;
}

.warning-panel {
    border: 1px solid #6d5a2d;
    background: #211c12;
    padding: 18px;
    color: #f0c96a;
}

.success-panel {
    border: 1px solid #285f50;
    background: #0e2821;
    padding: 18px;
    color: #45ffd2;
}

div[data-testid="stChatInput"] {
    border-top: 1px solid #202833;
}

button[kind="primary"] {
    border-radius: 2px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def clean_device(device):
    """
    Convert device representations such as cuda and cuda:0
    into a canonical torch.device.
    """
    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        return torch.device(device)

    return DEVICE


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def get_model_device(model):
    """Safely determine the device of a PyTorch model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return DEVICE


def extract_vocab_size(tokenizer):
    """
    Obtain vocabulary size without assuming a particular tokenizer
    attribute such as stoi.
    """

    possible_attributes = [
        "vocab_size",
        "vocabulary_size",
        "n_vocab",
        "num_tokens",
        "size",
    ]

    for attribute in possible_attributes:
        if hasattr(tokenizer, attribute):
            value = getattr(tokenizer, attribute)

            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue

            if isinstance(value, int):
                return value

    # Try common vocabulary containers.
    possible_containers = [
        "char_to_id",
        "id_to_char",
        "vocab",
        "vocabulary",
        "token_to_id",
        "id_to_token",
    ]

    for attribute in possible_containers:
        if hasattr(tokenizer, attribute):
            value = getattr(tokenizer, attribute)

            try:
                return len(value)
            except Exception:
                pass

    raise RuntimeError(
        "Unable to determine tokenizer vocabulary size."
    )


def tokenizer_encode(tokenizer, text):
    """
    Encode text using the project's tokenizer.

    Supports several common tokenizer APIs.
    """

    if not text:
        return []

    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text)

        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().tolist()

        return list(result)

    if hasattr(tokenizer, "tokenize"):
        tokens = tokenizer.tokenize(text)

        if hasattr(tokenizer, "convert_tokens_to_ids"):
            return tokenizer.convert_tokens_to_ids(tokens)

        return list(tokens)

    raise RuntimeError(
        "Tokenizer does not provide an encode() or tokenize() method."
    )


def tokenizer_decode(tokenizer, token_ids):
    """
    Decode token IDs using the project's tokenizer.
    """

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()

    token_ids = list(token_ids)

    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)

    if hasattr(tokenizer, "decode_ids"):
        return tokenizer.decode_ids(token_ids)

    if hasattr(tokenizer, "id_to_char"):
        mapping = getattr(tokenizer, "id_to_char")

        return "".join(
            mapping[int(token_id)]
            for token_id in token_ids
            if int(token_id) in mapping
        )

    raise RuntimeError(
        "Tokenizer does not provide a decode() or decode_ids() method."
    )


# ============================================================
# TOKENIZER LOADING
# ============================================================


@st.cache_resource
def load_tokenizer():
    """
    Load the project's CharacterTokenizer.

    This function deliberately avoids assuming that the tokenizer
    contains an attribute called 'stoi'.
    """

    from tokenizer.tokenizer import CharacterTokenizer

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAINING_DATA_PATH}"
        )

    text = TRAINING_DATA_PATH.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise RuntimeError(
            "training.txt exists but is empty."
        )

    # --------------------------------------------------------
    # Try the most common constructor forms.
    # --------------------------------------------------------

    attempts = []

    try:
        signature = inspect.signature(CharacterTokenizer)

        parameters = signature.parameters

        # Case 1:
        # CharacterTokenizer(text)
        if len(parameters) == 1:
            try:
                tokenizer = CharacterTokenizer(text)
                return tokenizer
            except Exception as error:
                attempts.append(str(error))

        # Case 2:
        # CharacterTokenizer()
        try:
            tokenizer = CharacterTokenizer()

            # Some implementations build vocabulary separately.
            if hasattr(tokenizer, "build_vocab"):
                tokenizer.build_vocab(text)

            elif hasattr(tokenizer, "fit"):
                tokenizer.fit(text)

            elif hasattr(tokenizer, "build_vocabulary"):
                tokenizer.build_vocabulary(text)

            return tokenizer

        except Exception as error:
            attempts.append(str(error))

    except Exception as error:
        attempts.append(str(error))

    # Final fallback attempts.
    try:
        return CharacterTokenizer(text=text)
    except Exception as error:
        attempts.append(str(error))

    try:
        return CharacterTokenizer(training_text=text)
    except Exception as error:
        attempts.append(str(error))

    raise RuntimeError(
        "Could not initialize CharacterTokenizer.\n\n"
        + "\n".join(attempts[-5:])
    )


# ============================================================
# MODEL LOADING
# ============================================================


def create_model(vocab_size):
    """
    Create the project's MiniGPT model.

    Supports the expected NEXUS constructor configuration.
    """

    from model.minigpt import MiniGPT

    # --------------------------------------------------------
    # First attempt: expected NEXUS constructor.
    # --------------------------------------------------------

    constructor_attempts = [
        {
            "vocab_size": vocab_size,
            "context_length": CONTEXT_LENGTH,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        {
            "vocab_size": vocab_size,
            "context_length": CONTEXT_LENGTH,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
        },
        {
            "vocab_size": vocab_size,
        },
    ]

    errors = []

    for kwargs in constructor_attempts:
        try:
            model = MiniGPT(**kwargs)
            return model
        except Exception as error:
            errors.append(
                f"{kwargs}: {error}"
            )

    # --------------------------------------------------------
    # Signature-based fallback.
    # --------------------------------------------------------

    try:
        signature = inspect.signature(MiniGPT)

        parameters = signature.parameters

        kwargs = {}

        if "vocab_size" in parameters:
            kwargs["vocab_size"] = vocab_size

        if "context_length" in parameters:
            kwargs["context_length"] = CONTEXT_LENGTH

        if "embedding_dim" in parameters:
            kwargs["embedding_dim"] = EMBEDDING_DIM

        if "num_heads" in parameters:
            kwargs["num_heads"] = NUM_HEADS

        if "num_layers" in parameters:
            kwargs["num_layers"] = NUM_LAYERS

        if "dropout" in parameters:
            kwargs["dropout"] = DROPOUT

        model = MiniGPT(**kwargs)

        return model

    except Exception as error:
        errors.append(
            f"Signature fallback: {error}"
        )

    raise RuntimeError(
        "Unable to initialize MiniGPT.\n\n"
        + "\n".join(errors)
    )


def extract_state_dict(checkpoint):
    """
    Extract model weights from several common checkpoint formats.
    """

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    possible_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "model_weights",
        "weights",
    ]

    for key in possible_keys:
        if key in checkpoint:

            value = checkpoint[key]

            if isinstance(value, dict):
                return value

            if isinstance(value, torch.nn.Module):
                return value.state_dict()

    # A raw state_dict itself is also a dictionary.
    if checkpoint:
        first_value = next(
            iter(checkpoint.values())
        )

        if isinstance(
            first_value,
            torch.Tensor
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find model weights in checkpoint."
    )


@st.cache_resource
def load_model():
    """
    Load tokenizer + checkpoint + model onto CUDA when available.
    """

    tokenizer = load_tokenizer()

    vocab_size = extract_vocab_size(tokenizer)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "NEXUS checkpoint not found.\n\n"
            f"Expected:\n{CHECKPOINT_PATH}\n\n"
            "Train the model first."
        )

    model = create_model(vocab_size)

    # --------------------------------------------------------
    # Load checkpoint safely.
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    # Remove optional DataParallel prefix.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load weights.
    # --------------------------------------------------------

    try:
        model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

    except RuntimeError as strict_error:

        # Try non-strict only to provide a better diagnostic.
        try:
            result = model.load_state_dict(
                cleaned_state_dict,
                strict=False,
            )

            if result.missing_keys:
                raise RuntimeError(
                    "Checkpoint/model mismatch.\n\n"
                    f"Missing keys: {result.missing_keys[:10]}"
                )

            if result.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint/model mismatch.\n\n"
                    f"Unexpected keys: "
                    f"{result.unexpected_keys[:10]}"
                )

        except Exception as fallback_error:

            raise RuntimeError(
                "NEXUS checkpoint could not be loaded.\n\n"
                f"Strict loading error:\n{strict_error}\n\n"
                f"Diagnostic:\n{fallback_error}"
            )

    # --------------------------------------------------------
    # Move model to GPU.
    # --------------------------------------------------------

    model = model.to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # Verify model device.
    # --------------------------------------------------------

    model_device = get_model_device(model)

    if DEVICE.type == "cuda":

        if model_device.type != "cuda":
            raise RuntimeError(
                "Model remained on CPU while CUDA was requested."
            )

        if model_device.index not in (None, 0):
            raise RuntimeError(
                f"Unexpected CUDA device: {model_device}"
            )

    return model, tokenizer, vocab_size


# ============================================================
# MODEL FORWARD
# ============================================================


def forward_model(model, input_ids):
    """
    Execute the model and extract logits.

    Supports models returning:
        logits
    or:
        logits, loss
    """

    result = model(input_ids)

    if isinstance(result, tuple):
        logits = result[0]

    elif isinstance(result, dict):
        if "logits" not in result:
            raise RuntimeError(
                "Model output dictionary does not contain 'logits'."
            )

        logits = result["logits"]

    else:
        logits = result

    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(
            "Model did not return a logits tensor."
        )

    return logits


# ============================================================
# SAMPLING
# ============================================================


def apply_repetition_penalty(
    logits,
    generated_ids,
    repetition_penalty,
):
    """
    Apply a standard repetition penalty to tokens that have
    already appeared.
    """

    if repetition_penalty <= 1.0:
        return logits

    unique_tokens = set(
        int(token_id)
        for token_id in generated_ids
    )

    for token_id in unique_tokens:

        if token_id < 0:
            continue

        if token_id >= logits.size(-1):
            continue

        token_logit = logits[..., token_id]

        if token_logit.item() < 0:
            logits[..., token_id] *= repetition_penalty
        else:
            logits[..., token_id] /= repetition_penalty

    return logits


def apply_top_k(logits, top_k):
    """
    Keep only the top-k token logits.
    """

    if top_k is None:
        return logits

    top_k = int(top_k)

    if top_k <= 0:
        return logits

    top_k = min(
        top_k,
        logits.size(-1),
    )

    values, _ = torch.topk(
        logits,
        top_k,
    )

    threshold = values[..., -1, None]

    logits = logits.masked_fill(
        logits < threshold,
        float("-inf"),
    )

    return logits


def apply_top_p(logits, top_p):
    """
    Nucleus sampling.
    """

    if top_p is None:
        return logits

    top_p = float(top_p)

    if top_p >= 1.0:
        return logits

    if top_p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    sorted_probabilities = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    sorted_remove = (
        cumulative_probabilities > top_p
    )

    sorted_remove[..., 1:] = (
        sorted_remove[..., :-1]
        .clone()
    )

    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(
        logits,
        dtype=torch.bool,
    )

    remove_mask.scatter_(
        -1,
        sorted_indices,
        sorted_remove,
    )

    logits = logits.masked_fill(
        remove_mask,
        float("-inf"),
    )

    return logits


# ============================================================
# GENERATION
# ============================================================


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
):
    """
    Autoregressive text generation.

    The implementation is intentionally kept here rather than
    depending on a specific generate() method signature.
    """

    encoded = tokenizer_encode(
        tokenizer,
        prompt,
    )

    if len(encoded) == 0:
        raise ValueError(
            "Prompt produced zero tokens."
        )

    input_ids = torch.tensor(
        encoded,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    generated_ids = input_ids.clone()

    model.eval()

    for _ in range(
        int(max_new_tokens)
    ):

        # ----------------------------------------------------
        # Respect context window.
        # ----------------------------------------------------

        context_ids = generated_ids[
            :, -CONTEXT_LENGTH:
        ]

        # ----------------------------------------------------
        # Forward pass.
        # ----------------------------------------------------

        logits = forward_model(
            model,
            context_ids,
        )

        if logits.ndim != 3:
            raise RuntimeError(
                "Expected logits shape [B, T, V]. "
                f"Received: {tuple(logits.shape)}"
            )

        # ----------------------------------------------------
        # Last-token logits.
        # ----------------------------------------------------

        next_token_logits = logits[
            :, -1, :
        ].clone()

        # ----------------------------------------------------
        # Repetition penalty.
        # ----------------------------------------------------

        next_token_logits = (
            apply_repetition_penalty(
                next_token_logits,
                generated_ids[0].tolist(),
                repetition_penalty,
            )
        )

        # ----------------------------------------------------
        # Temperature.
        # ----------------------------------------------------

        temperature = max(
            float(temperature),
            0.01,
        )

        next_token_logits = (
            next_token_logits / temperature
        )

        # ----------------------------------------------------
        # Top-K.
        # ----------------------------------------------------

        next_token_logits = apply_top_k(
            next_token_logits,
            top_k,
        )

        # ----------------------------------------------------
        # Top-P.
        # ----------------------------------------------------

        next_token_logits = apply_top_p(
            next_token_logits,
            top_p,
        )

        # ----------------------------------------------------
        # Probability distribution.
        # ----------------------------------------------------

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        # Numerical safety.
        if not torch.isfinite(
            probabilities
        ).all():

            probabilities = torch.softmax(
                logits[:, -1, :],
                dim=-1,
            )

        # ----------------------------------------------------
        # Sample next token.
        # ----------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # ----------------------------------------------------
        # Append.
        # ----------------------------------------------------

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token,
            ],
            dim=1,
        )

    # --------------------------------------------------------
    # Decode.
    # --------------------------------------------------------

    output = tokenizer_decode(
        tokenizer,
        generated_ids[0],
    )

    return output


# ============================================================
# SESSION STATE
# ============================================================


if "messages" not in st.session_state:
    st.session_state.messages = []


if "chat_count" not in st.session_state:
    st.session_state.chat_count = 1


if "last_inference_time" not in st.session_state:
    st.session_state.last_inference_time = 0.0


if "last_prompt_tokens" not in st.session_state:
    st.session_state.last_prompt_tokens = 0


if "last_response_tokens" not in st.session_state:
    st.session_state.last_response_tokens = 0


# ============================================================
# MODEL INITIALIZATION
# ============================================================


model = None
tokenizer = None
vocab_size = VOCAB_SIZE_DEFAULT
model_error = None


try:
    model, tokenizer, vocab_size = load_model()

except Exception as error:
    model_error = str(error)


# ============================================================
# MODEL INFORMATION
# ============================================================


if model is not None:

    try:
        total_parameters, trainable_parameters = (
            count_parameters(model)
        )

    except Exception:
        total_parameters = 0
        trainable_parameters = 0

    actual_model_device = get_model_device(
        model
    )

else:
    total_parameters = 0
    trainable_parameters = 0
    actual_model_device = DEVICE


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    st.markdown(
        '<div class="nexus-brand">NEXUS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nexus-subtitle">'
        'NEURAL INTELLIGENCE TERMINAL'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">SYSTEM</div>',
        unsafe_allow_html=True,
    )

    if model is not None:

        st.markdown(
            '<div class="status-online">'
            '<span class="status-dot">●</span> ONLINE'
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="status-offline">'
            '<span class="status-dot">●</span> OFFLINE'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-label">COMPUTE</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"DEVICE: {actual_model_device}"
    )

    st.caption(
        f"GPU: {GPU_NAME}"
    )

    st.caption(
        f"CUDA: {CUDA_VERSION}"
    )

    st.markdown(
        '<div class="section-label">MODEL</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "NAME: NEXUS-128"
    )

    st.caption(
        f"PARAMETERS: {total_parameters:,}"
    )

    st.caption(
        f"VOCABULARY: {vocab_size}"
    )

    st.caption(
        f"CONTEXT: {CONTEXT_LENGTH}"
    )

    st.caption(
        f"LAYERS: {NUM_LAYERS}"
    )

    st.caption(
        f"HEADS: {NUM_HEADS}"
    )

    st.markdown(
        '<div class="section-label">CHAT</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "+ NEW CHAT",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.chat_count += 1

        st.session_state.last_prompt_tokens = 0

        st.session_state.last_response_tokens = 0

        st.rerun()

    if st.session_state.messages:

        st.caption(
            f"CURRENT SESSION: "
            f"{st.session_state.chat_count:02d}"
        )

        st.caption(
            f"MESSAGES: "
            f"{len(st.session_state.messages)}"
        )

    else:

        st.caption(
            "NO CONVERSATION YET."
        )

    # --------------------------------------------------------
    # Generation settings.
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">GENERATION</div>',
        unsafe_allow_html=True,
    )

    temperature = st.slider(
        "TEMPERATURE",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.05,
    )

    max_tokens = st.slider(
        "MAX TOKENS",
        min_value=16,
        max_value=256,
        value=128,
        step=8,
    )

    top_k = st.slider(
        "TOP-K",
        min_value=0,
        max_value=86,
        value=40,
        step=1,
    )

    top_p = st.slider(
        "TOP-P",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
    )

    repetition_penalty = st.slider(
        "REPETITION PENALTY",
        min_value=1.0,
        max_value=1.5,
        value=1.1,
        step=0.01,
    )

    st.markdown(
        '<div class="section-label">ACTIONS</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "CLEAR CONVERSATION",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.last_prompt_tokens = 0

        st.session_state.last_response_tokens = 0

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================


st.markdown(
    f"""
<div class="terminal-header">
<div style="display:flex;justify-content:space-between;align-items:flex-end;">
<div>
<div class="terminal-title">NEXUS // NEURAL TERMINAL</div>
<div class="terminal-subtitle">A TRANSFORMER LANGUAGE MODEL BUILT FROM SCRATCH WITH PYTORCH</div>
</div>
<div class="terminal-meta">
<div><span class="meta-accent">●</span> {"ONLINE" if model is not None else "OFFLINE"}</div>
<div>MODEL: NEXUS-128</div>
<div>DEVICE: {actual_model_device}</div>
<div>GPU: {GPU_NAME}</div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL ERROR
# ============================================================


if model_error is not None:

    st.markdown(
        f"""
<div class="error-panel">
<div style="font-size:16px;margin-bottom:12px;">
NEXUS FAILED TO LOAD
</div>
<div style="white-space:pre-wrap;">
{model_error}
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL METRICS
# ============================================================


metric_1, metric_2, metric_3, metric_4 = st.columns(
    4
)


with metric_1:

    st.markdown(
        f"""
<div class="metric-box">
<div class="metric-title">PARAMETERS</div>
<div class="metric-value">{total_parameters:,}</div>
</div>
""",
        unsafe_allow_html=True,
    )


with metric_2:

    st.markdown(
        f"""
<div class="metric-box">
<div class="metric-title">VOCABULARY</div>
<div class="metric-value metric-accent">{vocab_size}</div>
</div>
""",
        unsafe_allow_html=True,
    )


with metric_3:

    st.markdown(
        f"""
<div class="metric-box">
<div class="metric-title">CONTEXT</div>
<div class="metric-value">{CONTEXT_LENGTH}</div>
</div>
""",
        unsafe_allow_html=True,
    )


with metric_4:

    st.markdown(
        f"""
<div class="metric-box">
<div class="metric-title">COMPUTE</div>
<div class="metric-value metric-accent">
{"CUDA" if DEVICE.type == "cuda" else "CPU"}
</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# ANALYTICS
# ============================================================


with st.expander(
    "SYSTEM / ANALYTICS"
):

    analytics_col1, analytics_col2 = st.columns(
        2
    )

    with analytics_col1:

        st.markdown(
            "### MODEL"

        )

        st.write(
            f"Model: **NEXUS-128**"
        )

        st.write(
            f"Parameters: **{total_parameters:,}**"
        )

        st.write(
            f"Trainable: **{trainable_parameters:,}**"
        )

        st.write(
            f"Layers: **{NUM_LAYERS}**"
        )

        st.write(
            f"Heads: **{NUM_HEADS}**"
        )

        st.write(
            f"Embedding dimension: **{EMBEDDING_DIM}**"
        )

    with analytics_col2:

        st.markdown(
            "### COMPUTE"
        )

        st.write(
            f"Device: **{actual_model_device}**"
        )

        st.write(
            f"GPU: **{GPU_NAME}**"
        )

        st.write(
            f"CUDA: **{CUDA_VERSION}**"
        )

        st.write(
            f"Context length: **{CONTEXT_LENGTH}**"
        )

        st.write(
            f"Vocabulary: **{vocab_size}**"
        )

    st.markdown(
        "### LAST INFERENCE"
    )

    st.write(
        f"Prompt tokens: "
        f"**{st.session_state.last_prompt_tokens}**"
    )

    st.write(
        f"Response tokens: "
        f"**{st.session_state.last_response_tokens}**"
    )

    st.write(
        f"Inference time: "
        f"**{st.session_state.last_inference_time:.2f}s**"
    )


# ============================================================
# CHAT HISTORY
# ============================================================


st.markdown(
    '<div class="section-label">NEURAL SESSION</div>',
    unsafe_allow_html=True,
)


if not st.session_state.messages:

    st.markdown(
        """
<div class="empty-terminal">
<div class="empty-title">
NEXUS // READY
</div>
<div class="empty-subtitle">
ENTER A MESSAGE TO BEGIN A NEURAL SESSION.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

else:

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        )

        if role == "user":

            st.markdown(
                f"""
<div class="chat-user">
<div class="chat-label">USER</div>
<div class="chat-text">{content}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
<div class="chat-nexus">
<div class="chat-label">NEXUS</div>
<div class="chat-text">{content}</div>
</div>
""",
                unsafe_allow_html=True,
            )


# ============================================================
# CHAT INPUT
# ============================================================


prompt = st.chat_input(
    "ENTER COMMAND OR MESSAGE..."
)


# ============================================================
# PROCESS PROMPT
# ============================================================


if prompt is not None:

    prompt = prompt.strip()

    if not prompt:

        st.warning(
            "Please enter a non-empty prompt."
        )

        st.stop()

    if model is None:

        st.error(
            "NEXUS is offline because the model could not be loaded."
        )

        st.stop()

    # --------------------------------------------------------
    # Store user message.
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # --------------------------------------------------------
    # Display user message immediately.
    # --------------------------------------------------------

    st.markdown(
        f"""
<div class="chat-user">
<div class="chat-label">USER</div>
<div class="chat-text">{prompt}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Generate response.
    # --------------------------------------------------------

    try:

        start_time = time.perf_counter()

        prompt_tokens = tokenizer_encode(
            tokenizer,
            prompt,
        )

        st.session_state.last_prompt_tokens = (
            len(prompt_tokens)
        )

        with st.spinner(
            "NEXUS // GENERATING..."
        ):

            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Remove the original prompt from generated text
        # if the tokenizer returned the complete sequence.
        # ----------------------------------------------------

        response_text = generated_text

        if generated_text.startswith(prompt):

            response_text = generated_text[
                len(prompt):
            ]

        response_text = response_text.strip()

        if not response_text:

            response_text = (
                "NEXUS generated an empty response."
            )

        # ----------------------------------------------------
        # Calculate response tokens.
        # ----------------------------------------------------

        try:

            response_tokens = tokenizer_encode(
                tokenizer,
                response_text,
            )

            st.session_state.last_response_tokens = (
                len(response_tokens)
            )

        except Exception:

            st.session_state.last_response_tokens = 0

        st.session_state.last_inference_time = (
            elapsed_time
        )

        # ----------------------------------------------------
        # Store response.
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
            }
        )

        # ----------------------------------------------------
        # Display response.
        # ----------------------------------------------------

        st.markdown(
            f"""
<div class="chat-nexus">
<div class="chat-label">NEXUS</div>
<div class="chat-text">{response_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.caption(
            f"NEXUS // "
            f"{elapsed_time:.2f}s // "
            f"PROMPT {st.session_state.last_prompt_tokens} TOKENS // "
            f"RESPONSE {st.session_state.last_response_tokens} TOKENS // "
            f"{actual_model_device}"
        )

    except Exception as error:

        error_message = str(error)

        st.markdown(
            f"""
<div class="error-panel">
<div style="font-size:14px;margin-bottom:8px;">
GENERATION ERROR
</div>
<div style="white-space:pre-wrap;">
{error_message}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Remove failed user message so the conversation
        # does not become inconsistent.
        if st.session_state.messages:

            if (
                st.session_state.messages[-1]["role"]
                == "user"
            ):

                st.session_state.messages.pop()


# ============================================================
# FOOTER
# ============================================================


st.markdown(
    """
<div style="
text-align:center;
color:#4e5968;
font-size:9px;
letter-spacing:2px;
margin-top:35px;
padding-top:15px;
border-top:1px solid #202833;
">
NEXUS • FROM-SCRATCH TRANSFORMER • PYTORCH • CUDA
</div>
""",
    unsafe_allow_html=True,
)