"""Association slider components."""
from typing import Optional

import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint
from typing_extensions import TypedDict

Gradient = TypedDict("Gradient", {"from": str, "to": str, "deg": int})


def create_slider(
    label: str,
    component_id: str,
    gradient: Optional[Gradient] = None,
    slider_color: str = "cyan",
) -> DashBlueprint:
    """Creates slider for the number of associated words.

    Parameters
    ----------
    label: str
        Label of the slider in a Badge.
    component_id: str
        Identifier of the component.
    gradient: Gradient or None
        Specifies gradient of the Badge.
    slider_color: str, default 'cyan'
        Color of the slider.

    Returns
    -------
    DashBlueprint
        Blueprint of the component.
    """
    association_slider = DashBlueprint()

    association_slider.layout = dmc.Grid(
        [
            dmc.Col(
                dmc.Badge(
                    label,
                    size="xl",
                    radius="xl",
                    variant="gradient",
                    gradient=gradient
                    or {"from": "teal", "to": "cyan", "deg": 105},
                ),
                span="content",
            ),
            dmc.Col(
                dmc.Slider(
                    id=component_id,
                    value=5,
                    min=0,
                    max=40,
                    step=1,
                    size="md",
                    radius="sm",
                    marks=[
                        {"value": value * 5, "label": f"{value*5}"}
                        for value in range(9)
                    ],
                    color=slider_color,
                    showLabelOnHover=False,
                ),
                span="auto",
            ),
        ],
    )
    return association_slider
