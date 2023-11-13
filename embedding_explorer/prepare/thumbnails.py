import string
from random import sample
from typing import Iterable, Optional

import numpy as np
import plotly.express as px
from wordcloud import WordCloud


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

COLORSCALES = px.colors.named_colorscales()


def create_random_scatter():
    n = 100
    x = np.random.rand(n)
    y = np.random.rand(n)
    colors = np.random.rand(n)
    sz = np.random.rand(n) * 30
    colorscale = COLORSCALES[np.random.randint(0, len(COLORSCALES))]
    fig = px.scatter(
        x=x,
        y=y,
        color=colors,
        size=sz,
        color_continuous_scale=colorscale,
        template="plotly_white",
    )
    fig.update_layout(width=800, height=600)
    fig.update_xaxes(zeroline=False, visible=False)
    fig.update_yaxes(zeroline=False, visible=False)
    fig.update_coloraxes(showscale=False)
    image_bytes = fig.to_image(format="svg")
    return image_bytes.decode("utf-8")


def generate_thumbnail(corpus: Optional[Iterable[str]]) -> str:
    """Generates thumbnail for given model and returns it as SVG string."""
    if corpus is None:
        return create_random_scatter()
    corpus_sample = sample(list(corpus), 200)
    corpus_joint = " ".join(corpus_sample)
    colormap = COLORMAPS[np.random.randint(0, len(COLORMAPS))]
    cloud = WordCloud(
        width=800, height=600, background_color="white", colormap=colormap
    )
    cloud.generate(corpus_joint)
    return cloud.to_svg()
