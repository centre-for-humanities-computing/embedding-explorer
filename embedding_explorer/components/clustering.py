"""Component code of the network graph."""
from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash_extensions.enrich import (
    DashBlueprint,
    Input,
    Output,
    State,
    dcc,
    exceptions,
)


def create_cluster_map(model_name: str) -> DashBlueprint:
    """Creates Network component blueprint."""
    network = DashBlueprint()

    network.layout = dcc.Graph(
        id=f"{model_name}_cluster_map",
        responsive=True,
        config={"scrollZoom": True},
        animation_options={"frame": {"redraw": True}},
        animate=False,
        className="h-full w-full",
    )

    @network.callback(
        Output(f"{model_name}_cluster_map", "figure"),
        Input(f"{model_name}_inference_data", "data"),
        prevent_initial_callback=True,
    )
    def update_network_figure(
        inference_data: Optional[Dict],
    ) -> go.Figure:
        """Updates the network when the selected words are changed."""
        if not inference_data:
            raise exceptions.PreventUpdate
        print("Updating figure")
        data = pd.DataFrame(inference_data)
        data["cluster_labels"] = data["cluster_labels"].map(
            str, na_action="ignore"
        )
        params = dict(x="x", y="y", color="cluster_labels")
        x_width = data.x.max() - data.x.min()
        y_width = data.y.max() - data.y.min()
        x_range = data.x.max() + (x_width * 0.25), data.x.min() - (
            x_width * 0.25
        )
        y_range = data.y.max() + (y_width * 0.25), data.y.min() - (
            y_width * 0.25
        )
        figure = px.scatter(data, **params)
        figure.update_layout(template="plotly_white", dragmode="pan")
        figure.update_xaxes(range=x_range)
        figure.update_yaxes(range=y_range)
        return figure

    return network
