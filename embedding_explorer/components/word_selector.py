"""Code for the word selector component."""
from typing import List

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import (
    DashBlueprint,
    Input,
    Output,
    State,
    dcc,
    exceptions,
)
from neofuzz import Process
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from typing_extensions import TypedDict


class Option(TypedDict):
    value: int
    label: str
    search: str


class DummyProcess:
    def query(self, search_terms, limit):
        return np.array([]), np.array([])


def create_word_selector(
    vocab: np.ndarray, model_name: str = "", fuzzy_search: bool = False
) -> DashBlueprint:
    """Creates word selector component blueprint."""

    word_selector = DashBlueprint()
    vocab_lookup = {word: index for index, word in enumerate(vocab)}
    if fuzzy_search:
        print("Indexing vocabulary for fuzzy search")
        vectorizer = make_pipeline(
            TfidfVectorizer(
                analyzer="char", ngram_range=(1, 5), max_features=50_000
            ),
            NMF(n_components=10),
        )
        fuzzy_process = Process(
            vectorizer,
            metric="euclidean",
            low_memory=True,
        )
        fuzzy_process.index(vocab)
        print("Indexing done")
    else:
        fuzzy_process = DummyProcess()

    word_selector.layout = dcc.Dropdown(
        # label="Seeds",
        # description="Select words that are going to be used as"
        # "the basis of semantic association.",
        id=f"{model_name}_word_selector",
        placeholder="Search for words...",
        value=[],
        # data=[],
        options=[],
        searchable=True,
        clearable=True,
        # size="lg",
        multi=True,
    )

    @word_selector.callback(
        Output(f"{model_name}_word_selector", "options"),
        Input(f"{model_name}_word_selector", "search_value"),
        State(f"{model_name}_word_selector", "value"),
    )
    def update_options(
        search_value: str, selected_values: List[int]
    ) -> List[Option]:
        if not search_value:
            raise exceptions.PreventUpdate
        # Collecting already chosen options
        selected_options = [
            dict(value=index, label=vocab[index], search=search_value)
            for index in selected_values
        ]
        # Lowercasing search value
        search_value = search_value.lower()
        # Trying to find exact match
        if search_value in vocab_lookup:
            exact_match = dict(
                value=vocab_lookup[search_value],
                label=search_value,
                search=search_value,
            )
            return [exact_match] + selected_options
        # Fuzzy finding 5 closest
        fuzzy_indices, _ = fuzzy_process.query(
            search_terms=[search_value], limit=5
        )
        fuzzy_indices = np.ravel(fuzzy_indices)
        # Collecting options
        fuzzy_matches = [
            dict(value=index, label=vocab[index], search=search_value)
            for index in fuzzy_indices
        ]
        return fuzzy_matches + selected_options

    return word_selector
