"""Blueprint for the main application."""
from typing import Any, List

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import DashBlueprint, dcc, html

from embedding_explorer.components.network import create_network
from embedding_explorer.components.slider import create_slider
from embedding_explorer.components.word_selector import create_word_selector


def create_blueprint(
    vocab: np.ndarray, embeddings: np.ndarray
) -> DashBlueprint:
    # --------[ Collecting blueprints ]--------
    word_selector = create_word_selector(vocab=vocab)
    first_level_slider = create_slider(
        component_id="first_level_association",
        slider_color="blue",
    )
    second_level_slider = create_slider(
        component_id="second_level_association",
        slider_color="indigo",
    )
    network = create_network(vocab=vocab, embeddings=embeddings)
    blueprints = [
        word_selector,
        first_level_slider,
        second_level_slider,
        network,
    ]

    # --------[ Creating app blueprint ]--------
    app_blueprint = DashBlueprint()
    app_blueprint.layout = html.Div(
        [
            html.Div(network.layout, className="flex-1 bg-red"),
            dmc.Stack(
                [
                    dmc.Center(),
                    dmc.Grid(
                        [
                            dmc.Col(word_selector.layout, span=6, offset=2),
                            dmc.Col(
                                dmc.Badge(
                                    "First level association:",
                                    size="xl",
                                    variant="gradient",
                                    fullWidth=True,
                                    gradient={"from": "indigo", "to": "blue"},
                                ),
                                span=1,
                                offset=2,
                            ),
                            dmc.Col(first_level_slider.layout, span=5),
                            dmc.Col(
                                dmc.Badge(
                                    "Second level association:",
                                    size="xl",
                                    variant="gradient",
                                    fullWidth=True,
                                    gradient={"from": "blue", "to": "indigo"},
                                ),
                                span=1,
                                offset=2,
                            ),
                            dmc.Col(second_level_slider.layout, span=5),
                        ],
                        gutter="lg",
                        columns=10,
                        align="stretch",
                    ),
                ],
                spacing="lg",
                className="p-8",
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
    return app_blueprint
