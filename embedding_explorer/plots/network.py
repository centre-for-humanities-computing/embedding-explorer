"""Plotting utilities for networks."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go


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
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode=display_mode,
        hoverinfo="text",
        text=labels or indices,
        textposition=textposition,
        marker={
            "color": color,
            "size": size,
            "colorbar": {"title": colorbar_title},
            "colorscale": "Rainbow",
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


def network_figure(
    node_x: np.ndarray,
    node_y: np.ndarray,
    edges: np.ndarray,
    node_labels: Optional[List[str]] = None,
    node_size: Union[np.ndarray, float] = 10.0,
    node_color: Union[np.ndarray, str] = "red",
    edge_weight: Union[np.ndarray, float] = 0.5,
    edge_color: Union[np.ndarray, str] = "#888",
    colorbar_title: str = "",
):
    """Produces network figure based on information about nodes and edges
    in a graph.

    Parameters
    ----------
    node_x: array of shape (n_nodes, )
        X position of nodes.
    node_y: array of shape (n_nodes, )
        Y position of nodes.
    edges: array of shape (n_edges, 2)
        Describes edges in form of pairs of node indices.
    node_labels: list of str or None, default None
        Labels to assign to each node, if not specified,
        node indices will be displayed.
    node_size: darray of shape (n_nodes,) or float, default 10
        Sizes of the nodes, if an array, different sizes will
        be used for each annotation.
    node_color: darray of shape (n_nodes,) or str, default "#ffb8b3"
        Specifies what color the nodes should be, if an array,
        different colors will be assigned to nodes based on a color scheme.
    edge_weight: darray of shape (n_edges,) or float, default 0.5
        Specifies the thickness of the edges connecting the nodes in the graph.
        If an array, different thicknesses will be used for each edge.
    edge_color: ndarray of shape (n-edges,) or str, default "#888"
        Specifies what color the edges should be, if an array,
        different colors will be assigned to edges based on a color scheme.

    Returns
    -------
    go.Figure
        Plotly figure of a network.
    """
    # Creating node trace for the network
    node_trace = create_node_trace(
        node_x,
        node_y,
        labels=node_labels,
        color=node_color,
        size=node_size,
        colorbar_title=colorbar_title,
    )
    # Creating edge trace lines
    edge_traces = create_edge_traces(
        node_x, node_y, edges, width=edge_weight, color=edge_color
    )
    # Making figure
    fig = go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            titlefont_size=16,
            showlegend=False,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig
