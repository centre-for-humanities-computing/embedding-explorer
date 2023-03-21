"""Blueprint for overview dashboard."""
from typing import Callable, Dict, List, Tuple
from urllib.parse import quote

import dash_mantine_components as dmc
from dash_extensions.enrich import Dash, DashBlueprint, dash, dcc, html

from embedding_explorer.blueprints.app import create_app
from embedding_explorer.components.model_card import create_card
from embedding_explorer.model import Model


def create_dashboard(models: Dict[str, Model]):
    """Creates dashboard for all embedding models.

    Parameters
    ----------
    models: dict of str to Model
        Mapping of names to models.
    """
    dashboard = DashBlueprint()

    # Collecting cards and registering pages
    cards = []
    pages = {}
    for model_name, model in models.items():
        cards.append(create_card(model=model, model_name=model_name))
        page = create_app(
            vocab=model.vocab,
            embeddings=model.embeddings,
            model_name=model_name,
        )
        page.register_callbacks(dashboard)
        pages[quote(model_name)] = page.layout

    dashboard.layout = html.Div(
        children=[
            html.Div(
                "Choose an embedding model to inspect:",
                className="text-2xl pt-8 pb-3 px-8",
            ),
            dmc.Grid(
                children=[dmc.Col(card.layout, span=1) for card in cards],
                gutter="lg",
                grow=True,
                columns=3,
                className="p-5",
            ),
        ]
    )

    main_blueprint = DashBlueprint()
    main_blueprint.layout = html.Div(dash.page_container)
    dashboard.register_callbacks(main_blueprint)

    def register_pages(app: Dash) -> None:
        dash.register_page("home", path="/", layout=dashboard.layout)
        for path, layout in pages.items():
            dash.register_page(path, layout=layout)

    return main_blueprint, register_pages
