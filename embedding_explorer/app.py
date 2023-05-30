import os
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Optional

from dash_extensions.enrich import Dash, DashBlueprint

from embedding_explorer.blueprints.dashboard import create_dashboard
from embedding_explorer.blueprints.explorer import create_explorer
from embedding_explorer.model import StaticEmbeddings


def get_dash_app(blueprint: DashBlueprint, **kwargs) -> Dash:
    """Returns app based on a blueprint with
    tailwindcss and font awesome added."""
    additional_scripts = kwargs.get("external_scripts", [])
    use_pages = kwargs.get("use_pages", False)
    pages_folder = "" if use_pages else "pages"
    app = Dash(
        blueprint=blueprint,
        title=kwargs.get("title", "embedding-explorer"),
        external_scripts=[
            {
                "src": "https://cdn.tailwindcss.com",
            },
            {
                "src": "https://kit.fontawesome.com/9640e5cd85.js",
                "crossorigin": "anonymous",
            },
            *additional_scripts,
        ],
        prevent_initial_callbacks=True,
        pages_folder=pages_folder,
        **kwargs,
    )
    return app


def is_notebook() -> bool:
    return "ipykernel" in sys.modules


def is_colab() -> bool:
    return "google.colab" in sys.modules


def open_url(url: str) -> None:
    if sys.platform == "win32":
        os.startfile(url)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", url])
    else:
        try:
            subprocess.Popen(["xdg-open", url])
        except OSError:
            print("Please open a browser on: " + url)


def run_silent(app: Dash, port: int) -> Callable:
    def _run_silent():
        import logging
        import warnings

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app.run_server(port=port)

    return _run_silent


def run_app(
    app: Dash,
    port: int = 8050,
) -> Optional[threading.Thread]:
    url = f"http://127.0.0.1:{port}/"
    if is_colab():
        from google.colab import output  # type: ignore

        thread = threading.Thread(target=run_silent(app, port))
        thread.start()
        time.sleep(4)

        print("Open in browser:")
        output.serve_kernel_port_as_window(
            port, anchor_text="Click this link to open topicwizard."
        )
        return thread

    elif is_notebook():
        from IPython.display import IFrame, display

        thread = threading.Thread(target=run_silent(app, port))
        thread.start()
        time.sleep(4)
        display(IFrame(src=url, width="1200", height="1000"))
        return thread
    else:
        open_url(url)
        app.run_server(port=port)


def show_explorer(
    model: StaticEmbeddings, port: int = 8050, fuzzy_search: bool = False
) -> Optional[threading.Thread]:
    """Visually inspect word embedding model with embedding-explorer.

    Parameters
    ----------
    model: StaticEmbeddings
        Named tuple of model vocabulary and matrix of embeddings.
    port: int
        Port for the app to run on.
    fuzzy_search: bool, default False
        Specifies whether you want to fuzzy search in the vocabulary.
        This is recommended for production use, but the index takes
        time to set up, therefore the startup time is expected to
        be greater.

    Returns
    -------
    Thread or None
        If the app runs in a Jupyter notebook, work goes on on
        a background thread, this thread is returned.
    """
    blueprint = create_explorer(model=model, fuzzy_search=fuzzy_search)
    app = get_dash_app(blueprint=blueprint, use_pages=False)
    return run_app(app, port=port)


def show_dashboard(
    models: Dict[str, StaticEmbeddings],
    port: int = 8050,
    fuzzy_search: bool = False,
) -> Optional[threading.Thread]:
    """Show dashboard for all given word embeddings.

    Parameters
    ----------
    models: dict of str to StaticEmbeddings
        Mapping of model names to models.
    fuzzy_search: bool, default False
        Specifies whether you want to fuzzy search in the vocabulary.
        This is recommended for production use, but the index takes
        time to set up, therefore the startup time is expected to
        be greater.

    port: int
        Port for the app to run on.

    Returns
    -------
    Thread or None
        If the app runs in a Jupyter notebook, work goes on on
        a background thread, this thread is returned.
    """
    blueprint, register_pages = create_dashboard(
        models=models, fuzzy_search=fuzzy_search
    )
    app = get_dash_app(blueprint=blueprint, use_pages=True)
    register_pages()
    return run_app(app, port=port)
