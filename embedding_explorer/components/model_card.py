import base64
from urllib.parse import quote

import dash_mantine_components as dmc
import numpy as np
from dash_extensions.enrich import DashBlueprint, html
from dash_iconify import DashIconify

from embedding_explorer.model import Model
from embedding_explorer.prepare.thumbnails import generate_thumbnail

COOL_ICONS = [
    "wind",
    "zap",
    "anchor",
    "cloud-drizzle",
    "cloud-snow",
    "compass",
    "dribble",
    "droplet",
    "gift",
    "meh",
    "moon",
    "smile",
    "sun",
    "truck",
    "umbrella",
]


def create_card(model: Model, model_name: str) -> DashBlueprint:
    """Creates card for model."""
    card = DashBlueprint()
    thumbnail = generate_thumbnail(model)
    encoded_thumbnail = base64.b64encode(thumbnail.encode("utf-8")).decode(
        "utf-8"
    )
    random_icon = COOL_ICONS[np.random.randint(0, len(COOL_ICONS))]
    card.layout = html.A(
        dmc.Card(
            children=[
                dmc.CardSection(
                    dmc.Image(
                        src="data:image/svg+xml;base64,{}".format(
                            encoded_thumbnail
                        ),
                    )
                ),
                dmc.CardSection(
                    [
                        html.Div(className="border-t-2 border-gray-300"),
                        dmc.Group(
                            [
                                DashIconify(
                                    icon=f"feather:{random_icon}", width=30
                                ),
                                html.Div(
                                    model_name,
                                    className="font-bold text-lg",
                                ),
                            ],
                            className="p-5",
                        ),
                    ],
                    className="""
                """,
                ),
            ],
            className="""
            rounded-xl border-solid border-2 border-gray-300
            bg-gradient-to-b from-white via-white to-gray-100
            hover:from-purple-100 hover:via-white hover:to-indigo-100
            transition-all duration-200
        """,
        ),
        href=quote(model_name),
    )
    return card
