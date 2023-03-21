"""Code for the word selector component."""
from typing import List

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    exceptions)
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
        search_value = search_value.lower()
        print(f"Updating options. {search_value}")
        if search_value in vocab_lookup:
            result = [
                {"value": vocab_lookup[search_value], "label": search_value}
            ]
            for index in selected_values:
                result.append({"value": index, "label": vocab[index]})
            return result
        matching_terms: List[Option] = [
            {"value": i_word, "label": word}
            for i_word, word in enumerate(vocab)
            if (search_value in word.lower()) or (i_word in selected_values)
        ]
        return matching_terms

    return word_selector
