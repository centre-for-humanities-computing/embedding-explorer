"""Plotting utilities for networks."""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import cyclical, sample_colorscale

from embedding_explorer.prepare.semkern import (SemanticKernel,
                                                calculate_n_connections,
                                                calculate_positions,
                                                get_closest_seed)


def _edge_pos(edges: np.ndarray, x_y: np.ndarray) -> np.ndarray:
    """
    Through a series of nasty numpy tricks, that I® wrote
    this function transforms edges and either the x or the y positions
    of nodes to the x or y positions for the lines in the plotly figure.
    In order for the line not to be connected, the algorithm
    has to insert a nan value after each pair of points
    that have to be connected.

    Parameters
    ----------
    edges: array of shape (n_edges, 2)
        Describes edges in form of pairs of node indices.
    x_y: array of shape (n_nodes,)
        X or Y coordinates of nodes.

    Returns
    -------
    X or Y coordinates of edges.
    """
    edges = np.array(edges)
    n_edges = edges.shape[0]
    x_y = np.array(x_y)
    # Get a view of positions where we have start and end positions
    # for each edge.
    end_node_positions = x_y[edges]
    # We pad the matrix with one more column
    padded = np.zeros((n_edges, 3))
    padded[:, :-1] = end_node_positions
    # That we fill up with NaNs
    padded[:, -1] = np.nan
    return padded.flatten()


def create_edge_trace(
    x: np.ndarray, y: np.ndarray, edges: np.ndarray
) -> go.Scatter:
    x_edges = _edge_pos(edges, x)
    y_edges = _edge_pos(edges, y)
    trace = go.Scatter(
        x=x_edges,
        y=y_edges,
        hoverinfo="none",
        mode="lines",
        showlegend=False,
        opacity=0.2,
        line=dict(color="black"),
    )
    return trace


def get_seed_colors(kernel: SemanticKernel) -> np.ndarray:
    """Returns array of RGB colors for each seed."""
    n_seeds = np.sum(kernel.priorities == 0)
    samplepoints = np.arange(n_seeds) / n_seeds
    colors = sample_colorscale(
        colorscale=cyclical.Phase, samplepoints=samplepoints
    )
    return np.array(colors)


def create_node_traces(
    kernel: SemanticKernel, x: np.ndarray, y: np.ndarray
) -> Tuple[go.Scatter, go.Scatter, go.Scatter]:
    """Creates node traces for the different levels of association."""
    closest_seed = get_closest_seed(kernel)
    scale = get_seed_colors(kernel)
    is_seed = kernel.priorities == 0
    sizes = calculate_n_connections(kernel.connections)
    sizes = (sizes / np.max(sizes)) * 60
    seed_trace = go.Scatter(
        name="",
        text=kernel.vocabulary[is_seed],
        x=x[is_seed],
        y=y[is_seed],
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            color=scale[closest_seed[is_seed]], size=sizes[is_seed], opacity=1
        ),
        textfont=dict(size=12, color="white"),
    )
    is_first_level = kernel.priorities == 1
    first_level_trace = go.Scatter(
        name="",
        text=kernel.vocabulary[is_first_level],
        x=x[is_first_level],
        y=y[is_first_level],
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            color=scale[closest_seed[is_first_level]],
            size=sizes[is_first_level],
            opacity=0.6,
        ),
    )
    is_second_level = kernel.priorities == 2
    second_level_trace = go.Scatter(
        name="",
        text=kernel.vocabulary[is_second_level],
        x=x[is_second_level],
        y=y[is_second_level],
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            color=scale[closest_seed[is_second_level]],
            size=sizes[is_second_level],
            opacity=0.4,
        ),
    )
    return second_level_trace, first_level_trace, seed_trace


def plot_semantic_kernel(kernel: SemanticKernel) -> go.Figure:
    """Plots semantic kernel."""
    x, y = calculate_positions(kernel.distance_matrix)

    node_traces = create_node_traces(x=x, y=y, kernel=kernel)
    edge_trace = create_edge_trace(x=x, y=y, edges=kernel.connections)

    figure = go.Figure(data=[edge_trace, *node_traces])
    figure.update_xaxes(
        showticklabels=False,
        title="",
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        linewidth=6,
        mirror=True,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    figure.update_yaxes(
        showticklabels=False,
        title="",
        gridcolor="#e5e7eb",
        linecolor="#f9fafb",
        mirror=True,
        linewidth=6,
        zerolinewidth=2,
        zerolinecolor="#d1d5db",
    )
    figure.update_layout(
        showlegend=False, paper_bgcolor="white", plot_bgcolor="white"
    )
    return figure
