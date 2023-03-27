from pathlib import Path

from embedding_explorer import show_explorer
from embedding_explorer.model import Model


def main():
    model = Model.load(Path("dat/google-news.npz"))
    show_explorer(model)


if __name__ == "__main__":
    main()
