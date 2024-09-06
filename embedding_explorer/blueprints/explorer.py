"""Blueprint for the main application."""

from typing import Iterable, Optional, Union

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import (DashBlueprint, Input, Output, State, dcc,
                                    html)
from neofuzz import Process
from sklearn.base import BaseEstimator

from embedding_explorer.components.network import create_network
from embedding_explorer.components.word_selector import create_word_selector


def create_explorer(
    corpus: Iterable[str],
    vectorizer: Optional[BaseEstimator] = None,
    embeddings: Optional[np.ndarray] = None,
    name: str = "",
    fuzzy_search: Union[bool, Process] = False,
) -> DashBlueprint:
    print(f"Creating explorer with name: {name}")
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
    word_selector = create_word_selector(
        corpus=corpus,
        vectorizer=vectorizer,
        fuzzy_search=fuzzy_search,
        model_name=name,
    )
    network = create_network(
        corpus=corpus,
        vectorizer=vectorizer,
        embeddings=embeddings,
        fuzzy_search=fuzzy_search,
        model_name=name,
    )
    blueprints = [
        word_selector,
        network,
    ]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            html.Div(network.layout, className="flex-1 bg-red"),
            html.Div(
                [
                    dmc.Text("Query parameters", size="lg", align="center"),
                    word_selector.layout,
                    dmc.Accordion(
                        chevronPosition="right",
                        variant="contained",
                        children=[
                            dmc.AccordionItem(
                                [
                                    dmc.AccordionControl(
                                        html.Div(
                                            [
                                                dmc.Text("Associations"),
                                                dmc.Text(
                                                    "Set the number of associations for each iteration",
                                                    size="sm",
                                                    weight=400,
                                                    color="dimmed",
                                                ),
                                            ]
                                        )
                                    ),
                                    dmc.AccordionPanel(
                                        [
                                            dmc.NumberInput(
                                                label="First level association",
                                                description="Number of closest words to find"
                                                "to the given seeds.",
                                                value=5,
                                                min=0,
                                                stepHoldDelay=500,
                                                stepHoldInterval=100,
                                                id="{}_first_level_association".format(
                                                    name
                                                ),
                                                size="md",
                                                className="mb-3",
                                            ),
                                            dmc.NumberInput(
                                                label="Second level association",
                                                description="Number of closest words to find to words"
                                                " found in the first level association.",
                                                value=5,
                                                min=0,
                                                stepHoldDelay=500,
                                                stepHoldInterval=100,
                                                id="{}_second_level_association".format(
                                                    name
                                                ),
                                                size="md",
                                            ),
                                        ]
                                    ),
                                ],
                                value="search",
                            ),
                        ],
                    ),
                    dmc.Accordion(
                        chevronPosition="right",
                        variant="contained",
                        children=[
                            dmc.AccordionItem(
                                [
                                    dmc.AccordionControl(
                                        html.Div(
                                            [
                                                dmc.Text("Export Settings"),
                                                dmc.Text(
                                                    "Set settings for image export.",
                                                    size="sm",
                                                    weight=400,
                                                    color="dimmed",
                                                ),
                                            ]
                                        )
                                    ),
                                    dmc.AccordionPanel(
                                        [
                                            dmc.NumberInput(
                                                label="Export Width",
                                                description="Width of the exported image.",
                                                value=1000,
                                                min=100,
                                                step=1,
                                                stepHoldDelay=500,
                                                stepHoldInterval=100,
                                                id="{}_export_width".format(
                                                    name
                                                ),
                                                size="md",
                                                className="mb-3",
                                            ),
                                            dmc.NumberInput(
                                                label="Export Height",
                                                description="Height of the exported image.",
                                                value=800,
                                                min=100,
                                                step=1,
                                                stepHoldDelay=500,
                                                stepHoldInterval=100,
                                                id="{}_export_height".format(
                                                    name
                                                ),
                                                size="md",
                                                className="mb-3",
                                            ),
                                            dmc.NumberInput(
                                                label="Scale",
                                                description="Factor to scale the image resolution with.",
                                                value=1,
                                                min=0.05,
                                                step=0.05,
                                                precision=2,
                                                id="{}_export_scale".format(
                                                    name
                                                ),
                                                size="md",
                                                className="mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                value="search",
                            ),
                        ],
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
                ],
                className="""
                fixed w-1/3 bottom-8 right-8
                bg-white shadow-2xl rounded-xl
                p-7 flex-col flex space-y-4
                hover:opacity-100 opacity-75
                transition-all duration-500
                """,
            ),
        ],
        className="""
            fixed w-full h-full flex-col flex items-stretch
            bg-white p-3 space-y-3
        """,
        id="words_container",
    )

    # --------[ Registering callbacks ]--------
    for blueprint in blueprints:
        blueprint.register_callbacks(app_blueprint)

    app_blueprint.clientside_callback(
        """
        function(width, height, scale) {
            const newConfig = {
                toImageButtonOptions: {height: height, width: width, scale: scale},
                scrollZoom: true,
            };
            return newConfig;
        }
        """,
        Output(f"{name}_network", "config"),
        Input(f"{name}_export_width", "value"),
        Input(f"{name}_export_height", "value"),
        Input(f"{name}_export_scale", "value"),
        prevent_initial_callback=False,
    )
    return app_blueprint
