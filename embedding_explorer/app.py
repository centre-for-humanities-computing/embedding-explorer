import dash_labs as dl
from dash_extensions.enrich import Dash, DashBlueprint


def get_dash_app(blueprint: DashBlueprint) -> Dash:
    """Returns app based on a blueprint with
    tailwindcss and font awesome added."""
    app = Dash(
        __name__,
        blueprint=blueprint,
        title="embedding-explorer",
        external_scripts=[
            {
                "src": "https://cdn.tailwindcss.com",
            },
            {
                "src": "https://kit.fontawesome.com/9640e5cd85.js",
                "crossorigin": "anonymous",
            },
        ],
        prevent_initial_callbacks=True,
        use_pages=True,
        pages_folder="",
    )
    return app
