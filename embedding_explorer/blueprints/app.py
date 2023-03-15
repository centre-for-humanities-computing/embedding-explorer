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
        label="First level association",
        component_id="first_level_association",
        gradient={"from": "cyan", "to": "teal", "deg": 105},
        slider_color="teal",
    )
    second_level_slider = create_slider(
        label="Second level association",
        component_id="second_level_association",
        gradient={"from": "teal", "to": "green", "deg": 105},
        slider_color="green",
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
            word_selector.layout,
            html.Div(network.layout, className="flex-1 bg-red"),
            dmc.Group(
                [
                    first_level_slider.layout,
                    second_level_slider.layout,
                ],
                grow=True,
                spacing=3,
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
