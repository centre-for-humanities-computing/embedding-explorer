from typing import Iterable, Optional

import dash_mantine_components as dmc
from dash_extensions.enrich import html


def create_clustering_settings(name):
    return [
        html.Div(
            [
                html.Div(
                    [
                        dmc.Text("Dimensionality Reduction"),
                        dmc.Text(
                            "Reduce dimensionality of embedding before clustering and projection.",
                            size="sm",
                            weight=400,
                            color="dimmed",
                        ),
                    ]
                ),
                dmc.Switch(
                    checked=False,
                    id="{}_dim_red_switch".format(name),
                ),
            ],
            className="w-full flex flex-row justify-between content-center",
        ),
        dmc.Group(
            [
                dmc.Select(
                    label="Select method",
                    description="Select dimensionality reduction method.",
                    id="{}_dim_red".format(name),
                    value="svd",
                    data=[
                        {
                            "value": "umap",
                            "label": "UMAP",
                        },
                        {
                            "value": "svd",
                            "label": "SVD",
                        },
                        {
                            "value": "nmf",
                            "label": "NMF",
                        },
                    ],
                    clearable=False,
                    className="hidden",
                ),
                dmc.NumberInput(
                    id="{}_n_dimensions".format(name),
                    label="Reduced Dimensionality",
                    description="Number of dimensions to reduce embeddings to.",
                    value=10,
                    min=2,
                    step=5,
                    className="hidden",
                ),
            ],
            grow=1,
        ),
        dmc.JsonInput(
            label="Additional Parameters",
            placeholder="Pass your additional parameters to the method here in JSON format.",
            formatOnBlur=True,
            autosize=True,
            minRows=2,
            id="{}_dim_red_params".format(name),
            className="hidden",
        ),
        html.Div(
            [
                html.Div(
                    [
                        dmc.Text("Clustering"),
                        dmc.Text(
                            "Cluster data points with unsupervised learning.",
                            size="sm",
                            weight=400,
                            color="dimmed",
                        ),
                    ]
                ),
                dmc.Switch(
                    checked=False,
                    id="{}_cluster_switch".format(name),
                ),
            ],
            className="w-full flex flex-row justify-between content-center",
        ),
        dmc.Group(
            [
                dmc.Select(
                    label="Select method",
                    description="Select clustering method.",
                    id="{}_clustering".format(name),
                    value="kmeans",
                    data=[
                        {
                            "value": "kmeans",
                            "label": "K-Means",
                        },
                        {
                            "value": "hdbscan",
                            "label": "HDBSCAN",
                        },
                        {
                            "value": "spectral",
                            "label": "Spectral",
                        },
                    ],
                    clearable=False,
                    className="hidden",
                ),
                dmc.NumberInput(
                    id="{}_n_clusters".format(name),
                    label="Cluster Number",
                    description="Number of clusters to find.",
                    value=10,
                    min=2,
                    step=1,
                    className="hidden",
                ),
            ],
            grow=1,
        ),
        dmc.JsonInput(
            label="Additional Parameters",
            placeholder="Pass your additional parameters to the method here in JSON format.",
            formatOnBlur=True,
            autosize=True,
            minRows=2,
            id="{}_clustering_params".format(name),
            className="hidden",
        ),
        html.Div(
            [
                dmc.Text("Projection"),
                dmc.Text(
                    "Choose a projection method to 2D space.",
                    size="sm",
                    weight=400,
                    color="dimmed",
                ),
            ]
        ),
        dmc.SegmentedControl(
            id="{}_projection_selector".format(name),
            value="umap",
            data=[
                {
                    "value": "umap",
                    "label": "UMAP",
                },
                {
                    "value": "tsne",
                    "label": "TSNE",
                },
                {
                    "value": "svd",
                    "label": "SVD",
                },
            ],
            fullWidth=True,
        ),
        dmc.JsonInput(
            label="Additional Parameters",
            placeholder="Pass your additional parameters to the projection method here in JSON format.",
            formatOnBlur=True,
            autosize=True,
            minRows=2,
            id="{}_projection_params".format(name),
        ),
        html.Button(
            "Submit",
            className="""
            rounded-xl text-white text-bold text-md
            p-3 mt-5
            transition-all duration-200 bg-gradient-to-bl
            from-cyan-500 via-blue-500 to-blue-400 bg-size-200
            hover:font-bold font-normal
            """,
            id="{}_submit_button".format(name),
        ),
    ]
