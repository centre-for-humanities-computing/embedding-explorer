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
    return padded  # .flatten()


def create_node_trace(
    x: np.ndarray,
    y: np.ndarray,
    labels: Optional[List[str]] = None,
    color: Union[np.ndarray, str] = "#ffb8b3",
    size: Union[np.ndarray, float] = 10,
    display_mode: str = "markers+text",
    textposition: Optional[str] = None,
    colorbar_title: str = "",
    max_size: int = 50,
) -> go.Scatter:
    """
    Draws a trace of nodes for a plotly network.

    Parameters
    ----------
    x: ndarray of shape (n_nodes,)
        x coordinates of nodes
    y: ndarray of shape (n_nodes,)
        y coordinates of nodes
    labels: list of str or None, default None
        Labels to assign to each node, if not specified,
        node indices will be displayed.
    size: ndarray of shape (n_nodes,) or float, default 10
        Sizes of the nodes, if an array, different sizes will
        be used for each annotation.
    color: ndarray of shape (n_nodes,) or str, default "#ffb8b3"
        Specifies what color the nodes should be, if an array,
        different colors will be assigned to nodes based on a color scheme.
    display_mode: str, default "markers"
        Specifies how the nodes should be displayed,
        consult Plotly documentation for further details
    textposition: str or None, default None
        Specifies how text should be positioned on the graph,
        consult Plotly documentation for further details
    max_size: int, default 20
        Maximal size of nodes.

    Returns
    ----------
    node_trace: go.Scatter
        Nodes for the graph drawn by plotly.
    """
    if not isinstance(size, float) and not isinstance(size, int):
        # Normalize size
        size_norm = np.linalg.norm(size, np.inf)
        size = (size / size_norm) * max_size + 5
    else:
        size = np.full(x.shape, size)
    indices = np.arange(len(x))
    if labels is None:
        labels = indices.tolist()
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode=display_mode,
        hoverinfo="text",
        text=labels,
        textposition=textposition,
        marker={
            "color": color,
            "size": size,
            "colorbar": {"title": colorbar_title},
            "colorscale": "Peach",
        },
        customdata=indices,
    )
    return node_trace  # type: ignore


def create_edge_traces(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    width: Union[np.ndarray, float] = 0.5,
    color: Union[np.ndarray, str] = "#888",
    max_width: int = 5,
) -> go.Figure:
    """
    Draws traces of edges for a plotly network.

    Parameters
    ----------
    x: ndarray of shape (n_nodes,)
        x coordinates of nodes
    y: ndarray of shape (n_nodes,)
        y coordinates of nodes
    edges: ndarray of shape (n_edges, 2) or None
        A matrix describing which nodes in the graph should be connected.
        Each row describes one connection with the indices of the two nodes.
        If not specified, a fully connected graph will be used.
    width: ndarray of shape (n_edges,) or float, default 0.5
        Specifies the thickness of the edges connecting the nodes in the graph.
        If an array, different thicknesses will be used for each edge.
    color: ndarray of shape (n-edges,) or str, default "#888"
        Specifies what color the edges should be, if an array,
        different colors will be assigned to edges based on a color scheme.
    max_width: int, default 5
        Maximum width of edges.

    Returns
    -------
    edge_trace: list of graph objects
        Edges for the graph drawn by plotly.
    """
    x_edges = _edge_pos(edges, x)
    y_edges = _edge_pos(edges, y)
    n_edges = edges.shape[0]
    indices = np.arange(n_edges)
    if isinstance(width, int) or isinstance(width, float):
        width = np.full(n_edges, width)
    else:
        size_norm = np.linalg.norm(width, 4)
        width = (width / size_norm) * max_width + 1
    if isinstance(color, str):
        color = np.full(n_edges, color)
    edge_trace = [
        go.Scatter(
            x=x_edges[i],
            y=y_edges[i],
            line={"width": width[i], "color": color[i]},
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )
        for i in indices
    ]
    return edge_trace  # type: ignore


def create_annotations(
    labels: List[str],
    x: np.ndarray,
    y: np.ndarray,
    size: Union[np.ndarray, float] = 5,
) -> List[Dict[str, Any]]:
    """
    Creates annotation objects for a plotly graph object.

    Parameters
    ----------
    labels: list of str
        List of annotation strings
    x: ndarray of shape (n_nodes,)
        x coordinates of annotations
    y: ndarray of shape (n_nodes,)
        y coordinates of annotations
    size: ndarray of shape (n_nodes,) or float, default 5
        Sizes of the annotations, if an array, different sizes will
        be used for each annotation.

    Returns
    ----------
    annotations: list of dict
        Plotly annotation objects
    """
    annotations = []
    if size is float:
        size = np.full(len(x), size)
    for i, label in enumerate(labels):
        annotations.append(
            dict(
                text=label,
                x=x[i],
                y=y[i],
                showarrow=False,
                xanchor="center",
                bgcolor="rgba(255,255,255,0.5)",
                bordercolor="rgba(0,0,0,0.5)",
                font={
                    "family": "Helvetica",
                    "size": size[i],
                    "color": "black",
                },
            )
        )
    return annotations


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
    edge_traces = create_edge_traces(x=x, y=y, edges=kernel.connections)

    figure = go.Figure(data=[*edge_traces, *node_traces])
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
    figure.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    return figure
