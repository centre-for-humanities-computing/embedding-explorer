"""Utilities for preparing Gensim word vectors for usage."""
from typing import Tuple

import numpy as np
from gensim.models import KeyedVectors


def prepare_keyed_vectors(
    keyed_vectors: KeyedVectors,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepares Gensim's KeyedVectors objects for usage in
    embedding-explorer.

    Parameters
    ----------
    keyed_vectors: KeyedVectors
        Gensim keyedvectors object.

    Returns
    -------
    vocab: array of str of shape (n_vocab, )
        Vocabulary of the model.
    embeddings: array of shape (n_vocab, vector_size)
        Word embeddings from the model.
    """
    vocab = np.array(keyed_vectors.index_to_key)
    embeddings = keyed_vectors.vectors
    return vocab, embeddings
