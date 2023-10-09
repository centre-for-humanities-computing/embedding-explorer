from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class Card(Mapping):
    """Contains information about an embedding model card that should be
    displayed on the dashboard.

    Parameters
    ----------
    corpus: iterable of string
        Texts you intend to search in with the semantic explorer.
    vectorizer: Transformer or None, default None
        Model to vectorize texts with.
        If not supplied the model is assumed to be a
        static word embedding model, and the embeddings
        parameter has to be supplied.
    embeddings: ndarray of shape (n_corpus, n_features), default None
        Embeddings of the texts in the corpus.
        If not supplied, embeddings will be calculated using
        the vectorizer.
    fuzzy_search: bool, default False
        Specifies whether you want to fuzzy search in the vocabulary.
        This is recommended for production use, but the index takes
        time to set up, therefore the startup time is expected to
        be greater.
    """

    name: str
    corpus: Iterable[str]
    vectorizer: Optional[BaseEstimator] = None
    embeddings: Optional[np.ndarray] = None
    fuzzy_search: bool = False

    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self):
        return ["name", "corpus", "vectorizer", "embeddings", "fuzzy_search"]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())
