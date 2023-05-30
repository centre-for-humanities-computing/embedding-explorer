"""Blueprint for overview dashboard."""
from typing import Callable, Dict, List, Tuple
from urllib.parse import quote

import dash_mantine_components as dmc
from dash_extensions.enrich import Dash, DashBlueprint, dash, dcc, html

from embedding_explorer.blueprints.explorer import create_explorer
from embedding_explorer.components.model_card import create_card


def create_dashboard(models: List[Dict], fuzzy_search: bool = False):
    """Creates dashboard for all static embedding models.

    Parameters
    ----------
    models: dict of str to StaticEmbeddings
        Mapping of names to models.
    """
    print("Creating Dashboard")
    dashboard = DashBlueprint()

    # Collecting cards and registering pages
    cards = []
    pages = {}
    for model_params in models:
        page = create_explorer(**model_params)
        cards.append(
            create_card(
                corpus=model_params["corpus"], model_name=model_params["name"]
            )
        )
        page.register_callbacks(dashboard)
        pages[model_params["name"]] = page.layout

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

    def register_pages():
        dash.register_page(
            "home", path="/", layout=dashboard.layout, redirect_from=["/home"]
        )
        for model_name, layout in pages.items():
            dash.register_page(
                quote(model_name), "/" + quote(model_name), layout=layout
            )

    return main_blueprint, register_pages
