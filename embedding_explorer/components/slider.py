"""Association slider components."""
from typing import Optional

import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint
from typing_extensions import TypedDict

Gradient = TypedDict("Gradient", {"from": str, "to": str, "deg": int})


def create_slider(
    component_id: str,
    slider_color: str = "cyan",
) -> DashBlueprint:
    """Creates slider for the number of associated words.

    Parameters
    ----------
    component_id: str
        Identifier of the component.
    slider_color: str, default 'cyan'
        Color of the slider.

    Returns
    -------
    DashBlueprint
        Blueprint of the component.
    """
    association_slider = DashBlueprint()

    association_slider.layout = dmc.Slider(
        id=component_id,
        value=5,
        min=0,
        max=40,
        step=1,
        size="md",
        radius="sm",
        marks=[
            {"value": value * 5, "label": f"{value*5}"} for value in range(9)
        ],
        color=slider_color,
        showLabelOnHover=False,
        p=8,
        mt=7,
    )
    return association_slider
