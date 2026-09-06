from pathlib import Path
from typing import List


class CharacterTokenizer:
    """
    A simple character-level tokenizer.

    It creates a vocabulary from text and provides
    methods for converting between characters and
    integer token IDs.
    """

    def __init__(self, text: str):
        """
        Build the vocabulary from the supplied text.

        Args:
            text: Training text used to construct the vocabulary.
        """

        if not text:
            raise ValueError("Training text cannot be empty.")

        # Find every unique character in the text.
        self.vocabulary = sorted(set(text))

        # Number of unique characters.
        self.vocab_size = len(self.vocabulary)

        # Character -> integer ID
        self.token_to_id = {
            token: index
            for index, token in enumerate(self.vocabulary)
        }

        # Integer ID -> character
        self.id_to_token = {
            index: token
            for index, token in enumerate(self.vocabulary)
        }

    def encode(self, text: str) -> List[int]:
        """
        Convert text into a list of integer token IDs.

        Example:
            "hello" -> [5, 2, 7, 7, 8]
        """

        token_ids = []

        for character in text:
            if character not in self.token_to_id:
                raise ValueError(
                    f"Character {character!r} is not in the vocabulary."
                )

            token_ids.append(self.token_to_id[character])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back into text.

        Example:
            [5, 2, 7, 7, 8] -> "hello"
        """

        characters = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(
                    f"Token ID {token_id} is not in the vocabulary."
                )

            characters.append(self.id_to_token[token_id])

        return "".join(characters)


def load_training_text(file_path: str) -> str:
    """
    Load training text from a file.

    Args:
        file_path: Path to the training text file.

    Returns:
        The contents of the file as a string.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training file was not found: {file_path}"
        )

    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    training_file = "data/raw/training.txt"

    text = load_training_text(training_file)

    tokenizer = CharacterTokenizer(text)

    print("=" * 50)
    print("NEXUS CHARACTER TOKENIZER")
    print("=" * 50)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    print("\nVocabulary:")
    print(tokenizer.vocabulary)

    test_text = "Hello"

    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print("\nTest text:")
    print(test_text)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(decoded)

    assert decoded == test_text

    print("\nTokenizer test PASSED.")