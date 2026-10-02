"""Data package: synthetic aligned image/text corpus generation and IO."""

from .loaders import load_corpus, save_corpus
from .synthetic import (
    generate_corpus,
    generate_train_test,
    list_concepts,
)

__all__ = [
    "generate_corpus",
    "generate_train_test",
    "list_concepts",
    "load_corpus",
    "save_corpus",
]
