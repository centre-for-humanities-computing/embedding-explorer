"""Blueprint for overview dashboard."""
from typing import List
from urllib.parse import quote

import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint, dash, html

from embedding_explorer.cards import Card, ClusteringCard
from embedding_explorer.components.model_card import create_card


def create_dashboard(cards: List[Card]):
    """Creates dashboard for all static embedding models.

    Parameters
    ----------
    models: list of Card
        Contains description of a model card that should appear in
        the dashboard.
    """
    print("Creating Dashboard")
    dashboard = DashBlueprint()

    # Collecting cards and registering pages
    card_components = []
    pages = {}
    for card in cards:
        page = card.get_page()
        if isinstance(card, ClusteringCard):
            corpus = None
        else:
            corpus = card["corpus"]
        card_components.append(
            create_card(corpus=corpus, model_name=card["name"])
        )
        page.register_callbacks(dashboard)
        pages[card["name"]] = page.layout

    dashboard.layout = html.Div(
        children=[
            html.Div(
                "Choose an embedding model to inspect:",
                className="text-2xl pt-8 pb-3 px-8",
            ),
            dmc.Grid(
                children=[
                    dmc.Col(card.layout, span=1) for card in card_components
                ],
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
