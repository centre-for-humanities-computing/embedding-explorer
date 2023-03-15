"""Code for the word selector component."""
from typing import List

import numpy as np
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    exceptions)
from typing_extensions import TypedDict


class Option(TypedDict):
    value: int
    label: str


def create_word_selector(vocab: np.ndarray) -> DashBlueprint:
    """Creates word selector component blueprint."""

    word_selector = DashBlueprint()

    word_selector.layout = dcc.Dropdown(
        id="word_selector",
        placeholder="Select words...",
        value=[],
        options=[],
        multi=True,
        searchable=True,
        className="min-w-max flex-1",
        clearable=True,
    )

    @word_selector.callback(
        Output("word_selector", "options"),
        Input("word_selector", "search_value"),
        State("word_selector", "value"),
    )
    def update_options(
        search_value: str, selected_values: List[Option]
    ) -> List[Option]:
        if not search_value:
            raise exceptions.PreventUpdate
        search_value = search_value.lower()
        print(f"Updating options. {search_value}")
        matching_terms: List[Option] = [
            {"value": i_word, "label": word}
            for i_word, word in enumerate(vocab)
            if (search_value in word.lower()) or (i_word in selected_values)
        ]
        return matching_terms

    return word_selector
