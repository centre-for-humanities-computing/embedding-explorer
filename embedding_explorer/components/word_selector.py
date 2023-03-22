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
from thefuzz import process
from typing_extensions import TypedDict


class Option(TypedDict):
    value: int
    label: str


def create_word_selector(
    vocab: np.ndarray, model_name: str = ""
) -> DashBlueprint:
    """Creates word selector component blueprint."""

    word_selector = DashBlueprint()
    vocab_lookup = {word: index for index, word in enumerate(vocab)}

    word_selector.layout = dmc.MultiSelect(
        label="Seeds",
        description="Select words that are going to be used as"
        "the basis of semantic association.",
        id=f"{model_name}_word_selector",
        placeholder="Search for words...",
        value=[],
        data=[],
        searchable=True,
        clearable=True,
        size="lg",
    )

    @word_selector.callback(
        Output(f"{model_name}_word_selector", "data"),
        Input(f"{model_name}_word_selector", "searchValue"),
        State(f"{model_name}_word_selector", "value"),
    )
    def update_options(
        search_value: str, selected_values: List[int]
    ) -> List[Option]:
        if not search_value:
            raise exceptions.PreventUpdate
        # Collecting already chosen options
        selected_options = [
            Option(value=index, label=vocab[index])
            for index in selected_values
        ]
        # Lowercasing search value
        search_value = search_value.lower()
        # Trying to find exact match
        if search_value in vocab_lookup:
            exact_match = Option(
                value=vocab_lookup[search_value], label=search_value
            )
            return [exact_match] + selected_options
        # Trying to fuzzy find 5 closest terms
        fuzzy_process_result = process.extract(search_value, vocab, limit=5)
        # Getting only the terms
        fuzzy_match_terms = [term for term, score in fuzzy_process_result]
        # Collecting options
        fuzzy_matches = [
            Option(value=vocab_lookup[term], label=term)
            for term in fuzzy_match_terms
        ]
        return fuzzy_matches + selected_options

    return word_selector
