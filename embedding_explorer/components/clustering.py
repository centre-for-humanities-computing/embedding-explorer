"""Component code of the network graph."""
from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    exceptions)


def create_cluster_map(
    model_name: str,
    metadata: Optional[pd.DataFrame],
    hover_name: Optional[str],
    hover_data,
) -> DashBlueprint:
    """Creates Network component blueprint."""
    network = DashBlueprint()

    network.layout = dcc.Graph(
        id=f"{model_name}_cluster_map",
        className="h-full w-full",
    )
    if metadata is not None:
        metadata = metadata.reset_index()

    @network.callback(
        Output(f"{model_name}_cluster_map", "figure"),
        Input(f"{model_name}_inference_data", "data"),
        Input(f"{model_name}_marker_color", "value"),
        Input(f"{model_name}_marker_size", "value"),
        Input(f"{model_name}_marker_label", "value"),
        Input(f"{model_name}_query", "data"),
        prevent_initial_callback=True,
    )
    def update_network_figure(
        inference_data: Optional[Dict],
        marker_color: str,
        marker_size: str,
        marker_label: str,
        query: str,
    ) -> go.Figure:
        """Updates the network when the selected words are changed."""
        if not inference_data:
            raise exceptions.PreventUpdate
        print("Updating figure")
        data = pd.DataFrame(inference_data)
        if metadata is not None:
            data = data.join(metadata)
        if query:
            data = data.query(query)
        data["cluster_labels"] = data["cluster_labels"].map(
            str, na_action="ignore"
        )
        params = dict(x="x", y="y", color=marker_color)
        if marker_size != "none":
            params["size"] = marker_size
        if marker_label != "none":
            params["text"] = marker_label
        if hover_name is not None:
            params["hover_name"] = hover_name
        if hover_data is not None:
            params["hover_data"] = hover_data
        figure = px.scatter(data, **params)
        figure.update_layout(
            template="plotly_white",
            margin=dict(l=0, r=10, b=0, t=0, pad=0),
        )
        figure.update_xaxes(
            title="",
        )
        figure.update_yaxes(
            title="",
        )
        return figure

    return network
