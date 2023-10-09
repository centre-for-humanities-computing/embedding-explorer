from typing import Iterable, TypedDict

import numpy as np
from sklearn.base import BaseEstimator


class Card(TypedDict, total=False):
    name: str
    corpus: Iterable[str]
    vectorizer: BaseEstimator
    embeddings: np.ndarray
    fuzzy_search: bool
