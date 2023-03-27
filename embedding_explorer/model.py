from pathlib import Path
from typing import NamedTuple

import numpy as np


class Model(NamedTuple):
    vocab: np.ndarray
    embeddings: np.ndarray

    @classmethod
    def from_keyed_vectors(cls, keyed_vectors) -> "Model":
        """Creates static embedding model from keyed vectors in Gensim."""
        vocab = np.array(keyed_vectors.index_to_key)
        embeddings = keyed_vectors.vectors
        return cls(vocab=vocab, embeddings=embeddings)

    def save(self, path: Path) -> None:
        """Saves static embedding model to the given path."""
        np.savez(path, vocab=self.vocab, embeddings=self.embeddings)

    @classmethod
    def load(cls, path: Path) -> "Model":
        """Loads static word embedding model from disk"""
        npzfile = np.load(path)
        return cls(vocab=npzfile["vocab"], embeddings=npzfile["embeddings"])
