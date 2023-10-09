"""Code for the word selector component."""
from typing import List, Optional, Union

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import (
    DashBlueprint,
    Input,
    Output,
    State,
    dcc,
    exceptions,
    html,
)
from neofuzz import Process
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from typing_extensions import TypedDict


class Option(TypedDict):
    value: Union[int, str]
    label: str
    search: str


class DummyProcess:
    def query(self, search_terms, limit):
        return np.array([]), np.array([])


def create_word_selector(
    corpus: np.ndarray,
    vectorizer: Optional[BaseEstimator],
    model_name: str = "",
    fuzzy_search: bool = False,
) -> DashBlueprint:
    """Creates word selector component blueprint."""
    word_selector = DashBlueprint()
    vocab_lookup = {word: index for index, word in enumerate(corpus)}
    if fuzzy_search:
        print("Indexing vocabulary for fuzzy search")
        if vectorizer is None:
            fuzzy_vectorizer = make_pipeline(
                TfidfVectorizer(
                    analyzer="char", ngram_range=(1, 4), max_features=20_000
                ),
                NMF(n_components=25),
            )
        else:
            fuzzy_vectorizer = vectorizer
        fuzzy_process = Process(
            fuzzy_vectorizer,
            metric="cosine",
            low_memory=True,
        )
        fuzzy_process.index(corpus)
        print("Indexing done")
    else:
        fuzzy_process = DummyProcess()

    search_bar = dcc.Dropdown(
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
    word_selector.layout = dmc.Accordion(
        chevronPosition="right",
        variant="contained",
        children=dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    html.Div(
                        [
                            dmc.Text("Seed words"),
                            dmc.Text(
                                "Words that will be used as the basis of association.",
                                size="sm",
                                weight=400,
                                color="dimmed",
                            ),
                        ]
                    ),
                ),
                dmc.AccordionPanel(search_bar),
            ],
            value="search",
        ),
    )

    @word_selector.callback(
        Output(f"{model_name}_word_selector", "options"),
        Input(f"{model_name}_word_selector", "search_value"),
        State(f"{model_name}_word_selector", "value"),
    )
    def update_options(
        search_value: str, selected_values: List[Union[int, str]]
    ) -> List[Option]:
        if not search_value:
            raise exceptions.PreventUpdate
        # Collecting already chosen options
        selected_options = [
            Option(
                value=value,
                label=corpus[value] if isinstance(value, int) else value,
                search=search_value,
            )
            for value in selected_values
        ]
        # Lowercasing search value
        search_value = search_value.lower()
        # Trying to find exact match
        if search_value in vocab_lookup:
            exact_match = Option(
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
            Option(value=index, label=corpus[index], search=search_value)
            for index in fuzzy_indices
        ]
        if vectorizer is not None:
            fuzzy_matches.append(
                Option(
                    value=search_value, label=search_value, search=search_value
                )
            )
        return fuzzy_matches + selected_options

    return word_selector
