from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from embedding_explorer import show_dashboard, show_explorer
from embedding_explorer.model import StaticEmbeddings


def main():
    kv = KeyedVectors.load("dat/glove-wiki-gigaword-50.gensim")
    model = StaticEmbeddings.from_keyed_vectors(kv)
    show_explorer(model=model, fuzzy_search=True)


if __name__ == "__main__":
    main()
