"""Blueprint for the main application."""
import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import DashBlueprint, dcc, html

from embedding_explorer.components.network import create_network
from embedding_explorer.components.word_selector import create_word_selector
from embedding_explorer.model import Model


def create_explorer(
    model: Model, model_name: str = "", fuzzy_search: bool = False
) -> DashBlueprint:
    # --------[ Collecting blueprints ]--------
    word_selector = create_word_selector(
        vocab=model.vocab, model_name=model_name, fuzzy_search=fuzzy_search
    )
    network = create_network(
        vocab=model.vocab, embeddings=model.embeddings, model_name=model_name
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
                    word_selector.layout,
                    dmc.NumberInput(
                        label="First level association",
                        description="Number of closest words to find"
                        "to the given seeds.",
                        value=5,
                        min=0,
                        stepHoldDelay=500,
                        stepHoldInterval=100,
                        id=f"{model_name}_first_level_association",
                        size="lg",
                    ),
                    dmc.NumberInput(
                        label="Second level association",
                        description="Number of closest words to find to words"
                        "found in the first level association.",
                        value=5,
                        min=0,
                        stepHoldDelay=500,
                        stepHoldInterval=100,
                        id=f"{model_name}_second_level_association",
                        size="lg",
                    ),
                    html.Button(
                        "Submit",
                        className="""
                        rounded-xl text-white text-bold text-lg
                        p-3 mt-5
                        transition-all duration-200 bg-gradient-to-bl
                        from-cyan-500 via-blue-500 to-blue-400 bg-size-200
                        hover:font-bold font-normal
                        """,
                        id=f"{model_name}_submit_button",
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
    return app_blueprint
