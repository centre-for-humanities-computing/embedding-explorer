import string

import numpy as np
from wordcloud import WordCloud

from embedding_explorer.model import Model


def is_alpha(word: str) -> bool:
    return all((c.isalpha() or c in string.printable for c in word))


COLORMAPS = [
    "PuBu",
    "BuPu",
    "PuBuGn",
    "BuGn",
    "PuRd",
    "Purples",
    "Blues",
]


def generate_thumbnail(model: Model) -> str:
    """Generates thumbnail for given model and returns it as SVG string."""
    vocab = model.vocab
    alphabetical = np.vectorize(is_alpha)(vocab)
    if vocab.shape[0] > 30:
        vocab = vocab[alphabetical]
    n_vocab = vocab.shape[0]
    n_top = min(n_vocab, 100)
    random_word_indices = np.random.randint(0, n_vocab, size=n_top)
    random_freqs = np.random.randint(0, 100, size=n_top)
    colormap = COLORMAPS[np.random.randint(0, len(COLORMAPS))]
    word_freqs = {
        vocab[index]: freq
        for index, freq in zip(random_word_indices, random_freqs)
    }
    cloud = WordCloud(
        width=800, height=600, background_color="white", colormap=colormap
    )
    cloud.generate_from_frequencies(word_freqs)
    return cloud.to_svg()
