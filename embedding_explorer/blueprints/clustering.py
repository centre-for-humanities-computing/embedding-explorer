"""Blueprint for the clustering application."""
import warnings
from typing import Iterable, Optional

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)
from dash_iconify import DashIconify
from sklearn.base import BaseEstimator

from embedding_explorer.components.clustering import create_cluster_map
from embedding_explorer.components.clustering_settings import \
    create_clustering_settings
from embedding_explorer.components.plot_settings import create_plot_settings
from embedding_explorer.prepare.clustering import (get_clustering,
                                                   get_projection,
                                                   get_reduced_embeddings)


def create_clustering_app(
    corpus: Optional[Iterable[str]] = None,
    vectorizer: Optional[BaseEstimator] = None,
    embeddings: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    name: str = "",
    hover_name: Optional[str] = None,
    hover_data=None,
) -> DashBlueprint:
    # Checking parameters
    if (embeddings is None) and (corpus is None and vectorizer is None):
        raise ValueError(
            "You either have to specify a corpus along with a vectorizer or an array of static embeddings."
        )
    if embeddings is None:
        embeddings = vectorizer.transform(corpus)
    if embeddings is not None and metadata is not None:
        if embeddings.shape[0] != len(metadata.index):
            raise ValueError(
                "Embeddings and metadata have to have the same length."
            )
    # --------[ Collecting blueprints ]--------
    cluster_map = create_cluster_map(name, metadata, hover_name, hover_data)
    clustering_settings = create_clustering_settings(name)
    plot_settings = create_plot_settings(name, metadata)
    blueprints = [cluster_map]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            html.Div(cluster_map.layout, className="flex-1 bg-red"),
            plot_settings,
            dcc.Store(id="{}_query".format(name), data=None),
            dcc.Store(id="{}_inference_data".format(name), data=None),
            dmc.Modal(
                title="Parameters",
                children=html.Div(
                    [
                        *clustering_settings,
                    ],
                    className="""
                    bg-white rounded p-4 flex-col flex space-y-4 items-stretch
                    w-full
                    """,
                ),
                zIndex=10000,
                centered=True,
                opened=True,
                id="{}_param_container".format(name),
            ),
            html.Button(
                [
                    "Open Parameters",
                    html.Div(className="w-3"),
                    DashIconify(icon="mingcute:settings-6-line", width=30),
                ],
                id="{}_open_params".format(name),
                className="""
                fixed bottom-8 right-8 shadow-md
                flex flex-row py-3 px-5 rounded-full
                bg-blue-500 hover:bg-blue-600 transition
                text-white items-center
                """,
            ),
        ],
        className="""
            fixed w-full h-full flex-col flex items-stretch p-8
        """,
        id="clustering_container",
    )
    app_blueprint.clientside_callback(
        """
        function(checked) {
            if(checked) {
                return ['visible', 'visible', 'visible']
            } else {
                return ['hidden', 'hidden', 'hidden']
            }
        }
        """,
        Output(f"{name}_dim_red", "className"),
        Output(f"{name}_dim_red_params", "className"),
        Output(f"{name}_n_dimensions", "className"),
        Input(f"{name}_dim_red_switch", "checked"),
    )
    app_blueprint.clientside_callback(
        """
        function(checked) {
            if(checked) {
                return ['visible', 'visible', 'visible']
            } else {
                return ['hidden', 'hidden', 'hidden']
            }
        }
        """,
        Output(f"{name}_clustering", "className"),
        Output(f"{name}_clustering_params", "className"),
        Output(f"{name}_n_clusters", "className"),
        Input(f"{name}_cluster_switch", "checked"),
    )
    app_blueprint.clientside_callback(
        """
        function(method) {
            if(method==='hdbscan') {
                return true
            } else {
                return false
            }
        }
        """,
        Output(f"{name}_n_clusters", "disabled"),
        Input(f"{name}_clustering", "value"),
    )
    app_blueprint.clientside_callback(
        """
        function(n1, n2, current) {
            return !current;
        }
        """,
        Output(f"{name}_param_container", "opened"),
        Input(f"{name}_submit_button", "n_clicks"),
        Input(f"{name}_open_params", "n_clicks"),
        State(f"{name}_param_container", "opened"),
        prevent_initial_callback=True,
    )

    @app_blueprint.callback(
        Output(f"{name}_query_input", "icon"),
        Output(f"{name}_query", "data"),
        Input(f"{name}_query_input", "value"),
    )
    def validate_query(query: str):
        if not query:
            return "", None
        try:
            metadata.head().query(query)
            return DashIconify(icon="mdi:tick", color="green"), query
        except Exception as e:
            warnings.warn(f"Failed query: {e}")
            return (
                DashIconify(
                    icon="material-symbols-light:mood-bad-outline",
                    color="red",
                ),
                None,
            )

    @app_blueprint.callback(
        Output(f"{name}_inference_data", "data"),
        Input(f"{name}_submit_button", "n_clicks"),
        State(f"{name}_cluster_switch", "checked"),
        State(f"{name}_clustering", "value"),
        State(f"{name}_n_clusters", "value"),
        State(f"{name}_clustering_params", "value"),
        State(f"{name}_dim_red_switch", "checked"),
        State(f"{name}_dim_red", "value"),
        State(f"{name}_n_dimensions", "value"),
        State(f"{name}_dim_red_params", "value"),
        State(f"{name}_projection_selector", "value"),
        State(f"{name}_projection_params", "value"),
    )
    def update_data(
        n_clicks: int,
        do_cluster: bool,
        clustering_method: str,
        n_clusters: int,
        clustering_params: str,
        do_dim_red: bool,
        dim_red_method: str,
        n_dimensions: int,
        dim_red_params: str,
        projection_method: str,
        projection_params: str,
    ):
        if not n_clicks:
            raise PreventUpdate()
        print("Inferring data")
        reduced_embeddings = get_reduced_embeddings(
            embeddings,
            do_dim_red,
            dim_red_method,
            n_dimensions,
            dim_red_params,
        )
        cluster_labels = get_clustering(
            reduced_embeddings,
            do_cluster,
            clustering_method,
            n_clusters,
            clustering_params,
        )
        x, y = get_projection(
            reduced_embeddings, projection_method, projection_params
        )
        return dict(cluster_labels=cluster_labels, x=x, y=y)

    # --------[ Registering callbacks ]--------
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)
    return app_blueprint
