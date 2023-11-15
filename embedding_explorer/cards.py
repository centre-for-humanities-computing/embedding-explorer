from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
from dash_extensions.enrich import DashBlueprint
from sklearn.base import BaseEstimator

from embedding_explorer.blueprints.clustering import create_clustering_app
from embedding_explorer.blueprints.explorer import create_explorer


class Card(ABC, Mapping):
    """Abstract Base Class for Cards in the Dashboard."""

    def __getitem__(self, key: str):
        return getattr(self, key)

    @abstractmethod
    def keys(self) -> List[str]:
        pass

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    @abstractmethod
    def get_page(self) -> DashBlueprint:
        """Get the Blueprint of the page for the card."""
        pass


@dataclass
class NetworkCard(Card):
    """Contains information about an embedding model card that should be
    displayed on the dashboard.
    This card will display the semantic network app when clicked on.

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

    def keys(self):
        return ["name", "corpus", "vectorizer", "embeddings", "fuzzy_search"]

    def get_page(self):
        return create_explorer(**self)


@dataclass
class ClusteringCard(Card):
    """Contains information about an embedding model card that should be
    displayed on the dashboard.
    This card will display the clustering network app when clicked on.

    Parameters
    ----------
    corpus: iterable of string, optional
        Texts you intend to cluster.
    vectorizer: TransformerMixin, optional
        Model to vectorize texts with.
    embeddings: ndarray of shape (n_corpus, n_features), optional
        Embeddings of the texts in the corpus.
        If not supplied, texts will be encoded with the vectorizer
    metadata: DataFrame, optional
        Metadata about the corpus or the embeddings.
        This is useful for filtering data points or
        changing visual properties of the main figure.
    hover_name: str, optional
        Title to display when hovering on a data point.
        Has to be the name of a column in the metadata.
    hover_data: list[str] or dict[str, bool], optional
        Additional data to display when hovering on a data point.
        Has to be a list of column names in the metadata,
        or a mapping of column names to booleans.
    port: int
        Port for the app to run on.

    """

    name: str
    corpus: Optional[Iterable[str]] = None
    vectorizer: Optional[BaseEstimator] = None
    embeddings: Optional[np.ndarray] = None
    metadata: Optional[pd.DataFrame] = None
    hover_name: Optional[str] = None
    hover_data: Any = None

    def keys(self):
        return [
            "name",
            "corpus",
            "embeddings",
            "vectorizer",
            "metadata",
            "hover_name",
            "hover_data",
        ]

    def get_page(self):
        return create_clustering_app(**self)
