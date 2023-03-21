from typing import NamedTuple

import numpy as np


class Model(NamedTuple):
    vocab: np.ndarray
    embeddings: np.ndarray
