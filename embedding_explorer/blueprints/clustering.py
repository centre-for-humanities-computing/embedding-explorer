"""Blueprint for the clustering application."""
from typing import Iterable, Optional

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import (
    DashBlueprint,
    Input,
    Output,
    State,
    dcc,
    html,
)
from sklearn.base import BaseEstimator

from embedding_explorer.components.clustering import create_cluster_map
from embedding_explorer.components.clustering_settings import (
    create_clustering_settings,
)
from embedding_explorer.prepare.clustering import (
    get_clustering,
    get_projection,
    get_reduced_embeddings,
)


def create_clustering_app(
    corpus: Iterable[str],
    vectorizer: Optional[BaseEstimator] = None,
    embeddings: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    name: str = "",
) -> DashBlueprint:
    # Checking parameters
    if embeddings is None and vectorizer is None:
        raise ValueError(
            "Either a vectorizer or static embeddings have to be supplied."
        )
    corpus = np.array(list(corpus))
    if (embeddings is not None) and (embeddings.shape[0] != corpus.shape[0]):
        raise ValueError(
            "The supplied corpus is not the same length"
            " as the embedding matrix."
        )
    if embeddings is None:
        embeddings = vectorizer.transform(corpus)
    # --------[ Collecting blueprints ]--------
    cluster_map = create_cluster_map(name)
    blueprints = [cluster_map]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            html.Div(cluster_map.layout, className="flex-1 bg-red"),
            dcc.Store(id="{}_inference_data".format(name), data=None),
            html.Div(
                [
                    dmc.Text("Parameters", size="lg", align="center"),
                    *create_clustering_settings(name),
                ],
                className="""
                fixed w-1/3 bottom-8 right-8
                bg-white shadow-2xl rounded-xl
                p-7 flex-col flex space-y-4
                hover:opacity-100 opacity-30
                transition-all duration-500
                """,
            ),
        ],
        className="""
            fixed w-full h-full flex-col flex items-stretch
            bg-white p-3 space-y-3
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
