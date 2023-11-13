import os
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from dash_extensions.enrich import Dash, DashBlueprint
from sklearn.base import BaseEstimator

from embedding_explorer.blueprints.clustering import create_clustering_app
from embedding_explorer.blueprints.dashboard import create_dashboard
from embedding_explorer.blueprints.explorer import create_explorer
from embedding_explorer.cards import Card


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
    else:
        open_url(url)
        app.run_server(port=port)


def show_network_explorer(
    corpus: Iterable[str],
    vectorizer: Optional[BaseEstimator] = None,
    embeddings: Optional[np.ndarray] = None,
    port: int = 8050,
    fuzzy_search: bool = False,
) -> Optional[threading.Thread]:
    """Visually inspect semantic networks emerging in an embedding model.

    Parameters
    ----------
    corpus: iterable of string
        Texts you intend to search in with the semantic explorer.
    vectorizer: Transformer or None, default None
        Model to vectorize texts with.
        If not supplied the model is assumed to be a
        static word embedding model, and the embeddings
        parameter has to be supplied.
    embeddings: ndarray of shape (n_corpus, n_features)
        Embeddings of the texts in the corpus.
        If not supplied, embeddings will be calculated using
        the vectorizer.
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
    blueprint = create_explorer(
        corpus=corpus,
        vectorizer=vectorizer,
        embeddings=embeddings,
        fuzzy_search=fuzzy_search,
    )
    app = get_dash_app(blueprint=blueprint, use_pages=False)
    return run_app(app, port=port)


def show_clustering(
    corpus: Optional[Iterable[str]] = None,
    vectorizer: Optional[BaseEstimator] = None,
    embeddings: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    hover_name: Optional[str] = None,
    hover_data=None,
    port: int = 8050,
) -> Optional[threading.Thread]:
    blueprint = create_clustering_app(
        corpus=corpus,
        vectorizer=vectorizer,
        embeddings=embeddings,
        metadata=metadata,
        hover_name=hover_name,
        hover_data=hover_data,
    )
    app = get_dash_app(blueprint=blueprint, use_pages=False)
    return run_app(app, port=port)


def show_dashboard(
    cards: List[Card],
    port: int = 8050,
) -> Optional[threading.Thread]:
    """Show dashboard for all given word embeddings.

    Parameters
    ----------
    cards: list of Card
        Contains description of a model card that should appear in
        the dashboard.

    port: int
        Port for the app to run on.

    Returns
    -------
    Thread or None
        If the app runs in a Jupyter notebook, work goes on on
        a background thread, this thread is returned.
    """
    if not all(["name" in kwargs for kwargs in cards]):
        raise ValueError(
            "You have to supply a 'name' attribute for each model."
        )
    blueprint, register_pages = create_dashboard(cards)
    app = get_dash_app(blueprint=blueprint, use_pages=True)
    register_pages()
    return run_app(app, port=port)
