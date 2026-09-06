NEXUS --- Neural Language Intelligence System

A small GPT-style language model and ChatGPT-like chatbot built from
scratch using Python and PyTorch.






Overview

NEXUS is an educational mini Large Language Model (LLM) built from
scratch in Python using PyTorch.

The goal of this project is not to compete with models such as ChatGPT,
but to understand the core architecture and workflow behind modern
autoregressive language models.

NEXUS implements the essential components of a small GPT-style
decoder-only Transformer:

Training
────────────────────────────────────────────

Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Training Sequences
   ↓
Token Embeddings
   ↓
Positional Embeddings
   ↓
Causal Self-Attention
   ↓
Multi-Head Attention
   ↓
Feed-Forward Network
   ↓
Layer Normalization
   ↓
Residual Connections
   ↓
Transformer Blocks
   ↓
Language Model Head
   ↓
Logits
   ↓
Cross-Entropy Loss
   ↓
Backpropagation
   ↓
AdamW
   ↓
Trained Model


Inference
────────────────────────────────────────────

User Prompt
   ↓
Tokenizer
   ↓
NEXUS
   ↓
Next-Token Prediction
   ↓
Sampling
   ↓
Autoregressive Generation
   ↓
Generated Response

Features

Custom tokenizer

Token-to-ID and ID-to-token conversion

Training dataset generation

Token embeddings

Positional embeddings

Self-attention

Causal masking

Multi-head attention

Feed-forward neural network

Residual connections

Layer normalization

Decoder-only Transformer architecture

Next-token prediction

Cross-entropy training

AdamW optimization

Training/validation loss tracking

Model checkpointing

Autoregressive text generation

Temperature-based sampling

Top-k sampling

Conversation history

Interactive chatbot

CPU/CUDA device detection

Streamlit web interface

Dark futuristic terminal-style UI

Model/system information display

Educational visualizations

Why NEXUS?

Most beginner LLM projects simply call an external API:

User → API → Existing LLM → Response

NEXUS takes a different approach:

User
 ↓
NEXUS Chatbot
 ↓
NEXUS Tokenizer
 ↓
NEXUS Transformer
 ↓
NEXUS Language Model
 ↓
Generated Response

The core model is trained from scratch so that the architecture can be
studied directly.

This makes NEXUS useful as a learning project for understanding:

Natural Language Processing

Deep Learning

Transformers

Attention mechanisms

Autoregressive language modeling

PyTorch

Model training

LLM inference

Architecture

1. Tokenization

Text cannot be directly processed by a neural network.

NEXUS first converts text into tokens and then into numerical IDs.

Example:

"hello"

      ↓

["h", "e", "l", "l", "o"]

      ↓

[7, 4, 9, 9, 12]

The tokenizer maintains a vocabulary that maps tokens to IDs.

token → ID
ID    → token

2. Token Embeddings

Token IDs are converted into dense vectors.

Token ID
   ↓
Embedding Lookup
   ↓
Vector Representation

For example:

"cat"
   ↓
[0.21, -0.17, 0.63, ...]

These vectors are learned during training.

3. Positional Embeddings

Transformers do not inherently understand the order of tokens.

NEXUS therefore adds positional information to token embeddings.

This allows the model to distinguish:

dog bites man

from:

man bites dog

4. Self-Attention

The attention mechanism allows each token to determine which previous
tokens are relevant.

NEXUS implements:

Q = XWq
K = XWk
V = XWv

and scaled dot-product attention:

Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V

Where:

Q = Query

K = Key

V = Value

dₖ = Key dimension

5. Causal Masking

Because NEXUS is an autoregressive language model, a token must not be
allowed to see future tokens during training.

A causal mask has a structure similar to:

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1

This ensures that predictions only depend on the current and previous
context.

6. Multi-Head Attention

Instead of using one attention mechanism, NEXUS uses multiple attention
heads.

Conceptually:

Input
  ↓
 ┌───────┬───────┬───────┬───────┐
 │ Head1 │ Head2 │ Head3 │ Head4 │
 └───────┴───────┴───────┴───────┘
  ↓
Concatenate
  ↓
Output Projection

Multiple heads allow the model to learn different relationships within
the context.

7. Feed-Forward Network

Each Transformer block also contains a feed-forward network:

Linear
  ↓
GELU
  ↓
Linear

The attention mechanism mixes information between tokens, while the
feed-forward network performs additional transformations on each token
representation.

8. Residual Connections

NEXUS uses residual connections around the major sublayers.

Conceptually:

x = x + attention_output
x = x + feed_forward_output

Residual connections help information and gradients flow through deeper
networks.

9. Layer Normalization

Layer normalization stabilizes the activations flowing through the
Transformer.

The Transformer block combines these components into a repeated
structure:

Input
  ↓
LayerNorm
  ↓
Causal Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual Connection
  ↓
Output

Model Configuration

The initial NEXUS configuration is intentionally small enough for
experimentation on consumer hardware.

Typical configuration:

Embedding Dimension : 128
Attention Heads     : 4
Transformer Layers  : 4
Context Length      : 128
Dropout             : 0.1

These values are configurable and can be increased depending on
available hardware and training data.

The project automatically detects whether CUDA is available:

torch.cuda.is_available()

and can use either:

CPU

or:

CUDA GPU

Training

NEXUS uses next-token prediction.

Given:

The cat sat on

the model attempts to predict the next token.

Training sequences are shifted by one position:

Input:
The cat sat on

Target:
cat sat on the

The model produces logits for possible next tokens.

These logits are compared against the correct target tokens using
cross-entropy loss.

The training process is:

Input Tokens
     ↓
Transformer
     ↓
Logits
     ↓
Cross Entropy Loss
     ↓
Backpropagation
     ↓
Gradients
     ↓
AdamW
     ↓
Parameter Update

Loss

NEXUS uses cross-entropy loss for next-token prediction.

The basic idea is:

Correct token probability ↑
        ↓
Loss ↓

and:

Correct token probability ↓
        ↓
Loss ↑

Training and validation loss can be monitored to identify learning and
potential overfitting.

Text Generation

After training, NEXUS can generate text autoregressively.

For example:

Prompt:
Artificial intelligence is

        ↓

Model predicts next token

        ↓

Artificial intelligence is a

        ↓

Model predicts another token

        ↓

Artificial intelligence is a field

        ↓

Continue...

Generation continues until the requested number of tokens has been
produced or an appropriate stopping condition is reached.

Sampling

NEXUS supports configurable generation behavior.

Temperature

Temperature controls randomness.

Low temperature
      ↓
More deterministic

High temperature
      ↓
More random

Typical values:

0.2  → conservative
0.7  → balanced
1.0  → more random
1.5  → highly random

Extremely high temperatures can produce incoherent output.

Top-k

Top-k sampling limits the next-token choices to the highest-scoring k
tokens before sampling.

Chatbot

NEXUS wraps the language model in a simple conversational interface.

Conversation history is maintained in a structure similar to:

[
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "What is AI?"}
]

The conversation is formatted into context and passed back into the
model during generation.

Streamlit Interface

NEXUS includes a Streamlit-based web interface designed around a
futuristic intelligence-terminal aesthetic.

The UI is inspired by the provided visual reference and uses:

Dark interface

Monospace typography

Thin technical borders

Terminal-style status indicators

System status panels

Model information

Chat history

Futuristic command input

Compact information windows

Cyberpunk/tactical dashboard styling

Conceptually:

┌───────────────────────────────────────────────────────────┐
│ NEXUS // NEURAL INTELLIGENCE TERMINAL     ● ONLINE       │
├──────────────┬────────────────────────────────────────────┤
│              │                                            │
│ NEXUS        │ USER                                       │
│              │ ─────────────────────────────              │
│ SYSTEM       │ Explain artificial intelligence             │
│ ● ONLINE     │                                            │
│              │ NEXUS                                      │
│ CHATS        │ ─────────────────────────────              │
│ 01 Research  │ Artificial intelligence is...              │
│ 02 Python    │                                            │
│ 03 AI        │                                            │
│              │                                            │
│ MODEL        │                                            │
│ NEXUS        │                                            │
│              │                                            │
│ SETTINGS     │ > ENTER MESSAGE...                         │
│              │ [SEND] [CLEAR]                             │
└──────────────┴────────────────────────────────────────────┘

Project Structure

NEXUS/
│
├── data/
│   ├── raw/
│   │   └── training.txt
│   └── processed/
│
├── tokenizer/
│   ├── __init__.py
│   └── tokenizer.py
│
├── model/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── feed_forward.py
│   ├── transformer_block.py
│   └── llm.py
│
├── training/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── inference/
│   ├── __init__.py
│   └── generate.py
│
├── chatbot/
│   ├── __init__.py
│   └── chat_engine.py
│
├── ui/
│   └── streamlit_app.py
│
├── utils/
│   ├── config.py
│   └── helpers.py
│
├── checkpoints/
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_attention.py
│   ├── test_model.py
│   └── test_generation.py
│
├── train.py
├── requirements.txt
├── README.md
└── .gitignore

Installation

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/NEXUS.git
cd NEXUS

Replace YOUR_USERNAME with your GitHub username.

2. Create a virtual environment

Windows

python -m venv .venv

Activate it:

.venv\Scripts\activate

Linux/macOS

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

Training NEXUS

Place the training corpus inside:

data/raw/training.txt

Then run:

python train.py

Training parameters can be adjusted through the project configuration.

The model should save its checkpoint inside:

checkpoints/

Running the Chatbot

After training and creating a checkpoint, run the chatbot/inference
entry point defined by the project.

For example:

python inference/generate.py

or use the Streamlit interface.

Running the Streamlit Application

Start the web application with:

streamlit run ui/streamlit_app.py

Streamlit will provide a local URL, typically similar to:

http://localhost:8501

Open that address in your browser.

Deployment

NEXUS can be deployed using Streamlit Community Cloud.

Basic deployment workflow:

Local NEXUS Project
        ↓
GitHub Repository
        ↓
Streamlit Community Cloud
        ↓
NEXUS Web Application

Before deploying:

Push the project to GitHub.

Ensure requirements.txt is present.

Configure the Streamlit entry point.

Make sure the model checkpoint is accessible to the deployment
environment.

Avoid committing large model files directly to Git if they exceed
repository limits.

For larger checkpoints, use an appropriate model/file hosting solution
or a smaller deployment checkpoint.

Important Limitations

NEXUS is an educational mini-LLM.

It is not equivalent to ChatGPT, Claude, Gemini, or other large
commercial language models.

Because the model is intentionally small and trained on a limited
dataset, it may:

Generate repetitive responses.

Produce grammatically incorrect text.

Hallucinate information.

Fail on unfamiliar questions.

Have very limited world knowledge.

Lose context in longer conversations.

Produce incomplete answers.

Generate nonsensical text.

These limitations are expected.

The purpose of NEXUS is to understand the architecture and training
process behind GPT-style language models.

What I Learned From This Project

Building NEXUS provides hands-on exposure to:

Machine Learning

Training loops

Loss functions

Optimization

Backpropagation

Validation

Overfitting

Deep Learning

Neural networks

Embeddings

Activations

Normalization

Residual connections

Parameter optimization

NLP

Tokenization

Vocabulary

Context windows

Language modeling

Next-token prediction

Transformers

Query / Key / Value

Self-attention

Scaled dot-product attention

Multi-head attention

Causal masking

Transformer blocks

LLM Inference

Autoregressive generation

Temperature

Top-k sampling

Context management

Conversation history

Deployment

Streamlit

Model loading

CPU/CUDA inference

Web application development

Future Improvements

Possible future versions of NEXUS can include:

NEXUS v1
  ↓
Character-level tokenizer
  ↓
NEXUS v2
  ↓
BPE tokenizer
  ↓
NEXUS v3
  ↓
Larger training corpus
  ↓
NEXUS v4
  ↓
Rotary Positional Embeddings (RoPE)
  ↓
NEXUS v5
  ↓
KV caching
  ↓
NEXUS v6
  ↓
Instruction fine-tuning
  ↓
NEXUS v7
  ↓
RAG + document understanding
  ↓
NEXUS v8
  ↓
Tool/function calling

Other possible improvements:

Better tokenization

Larger datasets

Larger Transformer architecture

Better training pipelines

GPU optimization

Mixed precision training

Flash Attention

KV caching

Quantization

LoRA/QLoRA

Instruction tuning

Retrieval-Augmented Generation

PDF/document question answering

Long-context support

Function calling

Agentic workflows

Educational Architecture Summary

                 ┌─────────────────────┐
                 │      Raw Text       │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │     Tokenizer       │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │    Token IDs        │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │     Embeddings      │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Positional Encoding │
                 └──────────┬──────────┘
                            ↓
          ┌─────────────────────────────────┐
          │       Transformer Blocks        │
          │                                 │
          │  LayerNorm                      │
          │      ↓                          │
          │  Multi-Head Causal Attention    │
          │      ↓                          │
          │  Residual Connection             │
          │      ↓                          │
          │  LayerNorm                      │
          │      ↓                          │
          │  Feed Forward                   │
          │      ↓                          │
          │  Residual Connection             │
          └────────────────┬────────────────┘
                           ↓
                 ┌─────────────────────┐
                 │    Language Head    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │       Logits        │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Next Token Sampling │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Generated Response  │
                 └─────────────────────┘

Project Status

Project: NEXUS
Type: Educational Mini LLM / Chatbot
Architecture: Decoder-only Transformer
Framework: PyTorch
Interface: Streamlit
Training: From scratch
Inference: Local
Deployment: Streamlit-compatible

Disclaimer

NEXUS is an educational implementation designed to demonstrate the
fundamental concepts behind GPT-style language models.

It should not be considered a production-grade conversational AI system.

The quality of generated responses depends heavily on:

Training data

Model size

Training duration

Hyperparameters

Tokenization

Hardware

Context length

Author

Built as a hands-on exploration of:

Python • PyTorch • NLP • Deep Learning • Transformers • LLMs •
Streamlit

The project focuses on understanding the internals of language models
rather than simply consuming an existing LLM API.

License

This project is intended for educational and research purposes.

Add an appropriate open-source license if you plan to distribute or
modify the project publicly.
