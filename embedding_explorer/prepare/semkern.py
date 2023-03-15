"""Prepares semantic kernels for the given seeds in a static word embedding model."""
from typing import List, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from networkx.drawing.layout import spring_layout
from sklearn.metrics import pairwise_distances


def closest_words(
    selected_words: List[int],
    embeddings: np.ndarray,
    n_closest: int,
    metric: str = "cosine",
) -> List[int]:
    """Returns words that are closely associated with the selected ones."""
    # Selecting terms
    selected_terms_matrix = embeddings[selected_words]
    # Calculating all distances from the selected words
    distances = pairwise_distances(
        selected_terms_matrix, embeddings, metric=metric
    )
    # Partitions array so that the smallest k elements along axis 1 are at the
    # lowest k dimensions, then I slice the array to only get the top indices
    # We do plus 1, as obviously the closest word is gonna be the word itself
    closest = np.argpartition(distances, kth=n_closest + 1, axis=1)[
        :, 1 : n_closest + 1
    ]
    associations = np.ravel(closest)
    association_set = set(associations) - set(selected_words)
    return list(association_set)


def calculate_dist_matrix(
    kernel_words: List[int], embeddings: np.ndarray
) -> np.ndarray:
    """Creates distance matrix of kernel words."""
    delta = pairwise_distances(embeddings[kernel_words])
    # Cut connections between the word and itself.
    np.fill_diagonal(delta, 0.0)
    # Cut connections that are over median distance
    delta[delta < np.median(delta)] = 0.0
    return delta


def semantic_kernel(
    seeds: List[int],
    embeddings: np.ndarray,
    vocab: np.ndarray,
    n_first_level: int,
    n_second_level: int,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates semantic kernel based on embeddings and seeds.

    Parameters
    ----------
    seeds: list of int
        Seeds to base the kernel on.
    embeddings: array of shape (n_vocab, n_dimensions)
        Word embedding matrix.
    vocab: array of str of shape (n_vocab, )
        Vocabulary of the embedding model.
    n_first_level: int
        Number of closest words to take from the seeds.
    n_second_level: int
        Number of closest words to take from the types.
    metric: str, default 'cosine'
        Distance metrics to use.

    Returns
    -------
    kernel_vocab: array of str of shape (n_kernel, )
        Vocabulary of the kernel.
    distance_matrix: array of shape (n_kernel, n_kernel)
        Distance matrix of the words in the kernel.
        Describes connections in the graph.
    """
    first_level = closest_words(
        selected_words=seeds,
        embeddings=embeddings,
        n_closest=n_first_level,
        metric=metric,
    )
    second_level = closest_words(
        selected_words=seeds + first_level,
        embeddings=embeddings,
        n_closest=n_second_level,
        metric=metric,
    )
    kernel_words = seeds + first_level + second_level
    seed_vocab = vocab[seeds].tolist()
    first_level_vocab = vocab[first_level].tolist()
    first_level_vocab = [word.upper() for word in first_level_vocab]
    second_level_vocab = vocab[second_level].tolist()
    kernel_vocab = seed_vocab + first_level_vocab + second_level_vocab
    return kernel_vocab, calculate_dist_matrix(
        kernel_words, embeddings=embeddings
    )


def calculate_positions(
    distance_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates node positions with Spring layout.

    Returns
    -------
    x: ndarray of shape (n_kernel,)
    y: ndarray of shape (n_kernel,)
    """
    positions = spring_layout(nx.from_numpy_matrix(distance_matrix))
    # This is some wizardry I honestly have no idea why I have to do this
    x, y = zip(*positions.values())
    return np.array(x), np.array(y)


def get_edges(distance_matrix: np.ndarray) -> np.ndarray:
    """Returns edges from the distance matrix."""
    graph = nx.from_numpy_matrix(distance_matrix)
    edges = np.array(graph.edges())
    return edges


def calculate_communities(distance_matrix: np.ndarray) -> np.ndarray:
    """Finds louvain communities in the graph and returns the community
    for each node."""
    graph = nx.from_numpy_matrix(distance_matrix)
    communities = louvain_communities(graph)
    community_per_node = np.empty(len(graph))
    for i_community, community in enumerate(communities):
        for node in community:
            community_per_node[node] = i_community
    return community_per_node


def calculate_n_connections(distance_matrix: np.ndarray) -> np.ndarray:
    """Calculates number of connections for each node in the graph."""
    connections = np.sum(distance_matrix != 0, axis=1)
    return connections
