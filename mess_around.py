from pathlib import Path
from random import sample

import pandas as pd
from embetter.text import SentenceEncoder

from embedding_explorer import show_dashboard, show_explorer


def main():
    data = pd.read_csv("abcnews-date-text.csv")
    corpus = data.headline_text.to_list()
    show_explorer(
        corpus=corpus,
        vectorizer=SentenceEncoder("all-mpnet-base-v2", device="cpu"),
    )


if __name__ == "__main__":
    main()
