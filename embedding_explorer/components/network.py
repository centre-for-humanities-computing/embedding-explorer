"""Component code of the network graph."""
from typing import List

import numpy as np
import plotly.graph_objects as go
from dash_extensions.enrich import (DashBlueprint, Input, Output, dcc,
                                    exceptions)

from embedding_explorer.plots.network import network_figure
from embedding_explorer.prepare.semkern import (calculate_communities,
                                                calculate_n_connections,
                                                calculate_positions, get_edges,
                                                semantic_kernel)


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
        kernel_vocab, distance_matrix = semantic_kernel(
            seeds=selected_words,
            embeddings=embeddings,
            vocab=vocab,
            n_first_level=n_first_level,
            n_second_level=n_second_level,
            metric="cosine",
        )
        node_x, node_y = calculate_positions(distance_matrix)
        edges = get_edges(distance_matrix)
        community_labels = calculate_communities(distance_matrix)
        n_connections = calculate_n_connections(distance_matrix)
        figure = network_figure(
            node_x=node_x,
            node_y=node_y,
            edges=edges,
            node_labels=kernel_vocab,
            node_size=n_connections,
            node_color=community_labels,
        )
        figure.update(layout_coloraxis_showscale=False)
        figure.update_traces(marker_showscale=False)
        return figure

    return network
