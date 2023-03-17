"""Component code of the network graph."""
from typing import List

import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import (DashBlueprint, Input, Output, dcc,
                                    exceptions)

from embedding_explorer.plots.network import plot_semantic_kernel
from embedding_explorer.prepare.semkern import create_semantic_kernel


def create_network(
    vocab: np.ndarray,
    embeddings: np.ndarray,
) -> DashBlueprint:
    """Creates Network component blueprint."""
    network = DashBlueprint()

    network.layout = dcc.Graph(
        id="network",
        responsive=True,
        config={"scrollZoom": True},
        animate=True,
        className="h-full w-full",
    )

    @network.callback(
        Output("network", "figure"),
        Input("word_selector", "value"),
        Input("first_level_association", "value"),
        Input("second_level_association", "value"),
        prevent_initial_callback=True,
    )
    def update_network_figure(
        selected_words: List[int], n_first_level: int, n_second_level: int
    ) -> go.Figure:
        """Updates the network when the selected words are changed."""
        if not selected_words:
            raise exceptions.PreventUpdate
        print("Redrawing network")
        kernel = create_semantic_kernel(
            seed_ids=selected_words,
            embeddings=embeddings,
            vocab=vocab,
            n_first_level=n_first_level,
            n_second_level=n_second_level,
        )
        figure = plot_semantic_kernel(kernel)
        figure.update(layout_coloraxis_showscale=False)
        figure.update_traces(marker_showscale=False)
        return figure

    return network
