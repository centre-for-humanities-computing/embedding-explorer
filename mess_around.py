from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from embedding_explorer import show_explorer
from embedding_explorer.model import Model


def main():
    kv = KeyedVectors.load("dat/glove-wiki-gigaword-50.gensim")
    model = Model.from_keyed_vectors(kv)
    # indices = np.random.choice(
    #     np.arange(len(model.vocab)), size=20_000, replace=False
    # )
    # model = Model(
    #     vocab=model.vocab[indices], embeddings=model.embeddings[indices]
    # )
    show_explorer(model)


if __name__ == "__main__":
    main()
