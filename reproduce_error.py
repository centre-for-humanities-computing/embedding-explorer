from typing import Iterable, Optional

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


def edge_pos(pos: Iterable[float]) -> list[Optional[float]]:
    res = []
    for start_pos in pos:
        for end_pos in pos:
            res.extend([start_pos, end_pos, None])
    return res


app = Dash()

app.layout = html.Div(
    [
        html.Button(id="button", children="Click me!", n_clicks=0),
        dcc.Graph(
            id="figure",
            style={"flex": 1},
        ),
    ],
    style={
        "display": "flex",
        "position": "fixed",
        "width": "100%",
        "height": "100%",
    },
)


@app.callback(Output("figure", "figure"), Input("button", "n_clicks"))
def update_fig(n_clicks: int) -> go.Figure:
    # Randomly generate points from a standard normal
    x, y = np.random.normal(0, 1, size=(2, 5))
    # Create the trace for nodes
    node_trace = go.Scatter(x=x, y=y, mode="markers")
    # Create a trace of lines connecting them
    n_clicks = n_clicks / 100
    edge_trace = go.Scatter(
        name=str(np.random.normal(0, 1)),
        x=[
            0,
            np.cos(n_clicks) - np.sin(n_clicks),
            None,
            0,
            -np.cos(n_clicks) - np.sin(n_clicks),
            None,
        ],
        y=[
            0,
            np.cos(n_clicks) + np.sin(n_clicks),
            None,
            0,
            np.cos(n_clicks) - np.sin(n_clicks),
            None,
        ],
        mode="lines",
    )
    traces = [edge_trace]
    # if n_clicks % 2:
    #     traces.append(edge_trace)
    fig = go.Figure(
        data=[*traces],
    )
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-1, 1])
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
