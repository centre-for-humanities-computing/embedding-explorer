"""Plotting utilities for networks."""
from typing import Tuple

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


def minmax(a: np.ndarray) -> np.ndarray:
    """Min-max normalizes an array."""
    return (a - np.min(a)) / (np.max(a) - np.min(a))


def add_edges(
    fig: go.Figure,
    edges: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    distance_matrix: np.ndarray,
) -> go.Figure:
    """Adds edges to a figure as shapes."""
    opacities = np.array(
        [-distance_matrix[start, end] for start, end in edges]
    )
    opacities = minmax(opacities + 1)  # / 1.5
    for (start, end), opacity in zip(edges, opacities):
        distance = distance_matrix[start, end]
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x[start],
            y0=y[start],
            x1=x[end],
            y1=y[end],
            label=dict(text=f"{distance:.2f}", font=dict(size=12)),
            layer="below",
            opacity=opacity,
            line=dict(width=3),
        )
    return fig


def get_seed_colors(kernel: SemanticKernel) -> np.ndarray:
    """Returns array of RGB colors for each seed."""
    n_seeds = np.sum(kernel.priorities == 0)
    samplepoints = np.arange(n_seeds) / n_seeds
    colors = sample_colorscale(
        colorscale=cyclical.Phase, samplepoints=samplepoints
    )
    return np.array(colors)


def add_nodes(
    fig: go.Figure, kernel: SemanticKernel, x: np.ndarray, y: np.ndarray
) -> go.Figure:
    """Creates node traces for the different levels of association."""
    closest_seed = get_closest_seed(kernel)
    scale = get_seed_colors(kernel)
    is_seed = kernel.priorities == 0
    sizes = calculate_n_connections(kernel.connections)
    sizes = (sizes / np.max(sizes)) * 100
    annotations = []
    seed_trace = go.Scatter(
        name="",
        x=x[is_seed],
        y=y[is_seed],
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            color=scale[closest_seed[is_seed]],
            size=sizes[is_seed],
            opacity=1,
            line=dict(width=3, color="black"),
        ),
    )
    for node_x, node_y, text, color in zip(
        x[is_seed],
        y[is_seed],
        kernel.vocabulary[is_seed],
        scale[closest_seed[is_seed]],
    ):
        annotations.append(
            dict(
                x=node_x,
                y=node_y,
                text=f"<b>{text.upper()}</b>",
                bgcolor=color,
                font=dict(size=18, color="white"),
                align="center",
                borderpad=4,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                showarrow=False,
                opacity=0.9,
            )
        )
    is_first_level = kernel.priorities == 1
    first_level_trace = go.Scatter(
        name="",
        x=x[is_first_level],
        y=y[is_first_level],
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            color=scale[closest_seed[is_first_level]],
            size=sizes[is_first_level],
            line=dict(width=2, color="#0C090A"),
            opacity=1,
        ),
    )
    for node_x, node_y, text, color in zip(
        x[is_first_level],
        y[is_first_level],
        kernel.vocabulary[is_first_level],
        scale[closest_seed[is_first_level]],
    ):
        annotations.append(
            dict(
                x=node_x,
                y=node_y,
                text=f"<b>{text}</b>",
                bgcolor=color,
                font=dict(size=16, color="white"),
                align="center",
                borderpad=4,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                showarrow=False,
                opacity=0.9,
            )
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
            opacity=1,
            line=dict(width=2, color="#0C090A"),
        ),
        textfont=dict(size=16, color=scale[closest_seed[is_second_level]]),
        textposition="bottom center",
    )
    fig.add_trace(second_level_trace)
    fig.add_trace(first_level_trace)
    fig.add_trace(seed_trace)
    for annotation in annotations[::-1]:
        fig.add_annotation(**annotation)
    return fig


def plot_semantic_kernel(kernel: SemanticKernel) -> go.Figure:
    """Plots semantic kernel."""
    x, y = calculate_positions(kernel.distance_matrix)

    figure = go.Figure()
    add_nodes(figure, kernel, x, y)
    add_edges(
        figure,
        edges=kernel.connections,
        x=x,
        y=y,
        distance_matrix=kernel.distance_matrix,
    )
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
