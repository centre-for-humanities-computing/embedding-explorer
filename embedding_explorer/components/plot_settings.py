import dash_mantine_components as dmc
import pandas as pd
from dash_extensions.enrich import html


def create_plot_settings(name: str, metadata: pd.DataFrame):
    if metadata is None:
        visibility = "hidden"
        columns = []
        numeric_columns = []
    else:
        visibility = "visible"
        columns = metadata.columns
        numeric_columns = metadata.select_dtypes(include="number").columns
    return html.Div(
        children=[
            dmc.Select(
                label="Marker color",
                description="Choose which property of the data points determines their color.",
                value="cluster_labels",
                id="{}_marker_color".format(name),
                data=[
                    {
                        "value": "cluster_labels",
                        "label": "Cluster Labels",
                    },
                ]
                + [{"value": column, "label": column} for column in columns],
            ),
            dmc.Select(
                label="Marker size",
                description="Choose which property of the data points determines their size.",
                value="none",
                id="{}_marker_size".format(name),
                data=[
                    {
                        "value": "none",
                        "label": "None",
                    },
                ]
                + [
                    {"value": column, "label": column}
                    for column in numeric_columns
                ],
            ),
            dmc.Select(
                label="Marker label",
                description="Choose which property of the data points determines their label.",
                value="none",
                id="{}_marker_label".format(name),
                data=[
                    {
                        "value": "none",
                        "label": "None",
                    },
                ]
                + [{"value": column, "label": column} for column in columns],
            ),
            dmc.TextInput(
                id="{}_query_input".format(name),
                label="Filter",
                description="Specify pandas query by which the data points should be filtered.",
                placeholder="Write something like: (genre == 'Tragedy') & (length > 100)",
                styles={
                    "input": {
                        "font-family": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
                    }
                },
                icon="",
            ),
        ],
        className="""
        flex-row w-full flex justify-start items-center content-center
        space-x-8 grow-0 shrink p-5 max-h-max
        """
        + visibility,
    )
