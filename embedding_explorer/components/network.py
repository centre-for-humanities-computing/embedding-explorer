"""Component code of the network graph."""

from typing import List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    exceptions)
from neofuzz import Process
from sklearn.base import BaseEstimator

from embedding_explorer.plots.network import plot_semantic_kernel
from embedding_explorer.prepare.semkern import create_semantic_kernel


def create_network(
    corpus: np.ndarray,
    embeddings: np.ndarray,
    vectorizer: Optional[BaseEstimator],
    model_name: str = "",
    fuzzy_search: Union[bool, Process] = False,
) -> DashBlueprint:
    """Creates Network component blueprint."""
    network = DashBlueprint()

    network.layout = dcc.Graph(
        id=f"{model_name}_network",
        responsive=True,
        config={"scrollZoom": True},
        animation_options={"frame": {"redraw": True}},
        animate=False,
        className="h-full w-full",
    )

    @network.callback(
        Output(f"{model_name}_network", "figure"),
        Input(f"{model_name}_submit_button", "n_clicks"),
        State(f"{model_name}_word_selector", "value"),
        State(f"{model_name}_first_level_association", "value"),
        State(f"{model_name}_second_level_association", "value"),
        prevent_initial_callback=True,
    )
    def update_network_figure(
        n_clicks: int,
        selected_words: List[Union[int, str]],
        n_first_level: int,
        n_second_level: int,
    ) -> go.Figure:
        """Updates the network when the selected words are changed."""
        if not selected_words or not n_clicks:
            raise exceptions.PreventUpdate
        kernel = create_semantic_kernel(
            selected=selected_words,
            embeddings=embeddings,
            corpus=corpus,
            vectorizer=vectorizer,
            n_first_level=n_first_level,
            n_second_level=n_second_level,
        )
        figure = plot_semantic_kernel(kernel)
        # figure.show()
        figure.update_layout(dragmode="pan")
        return figure

    return network
