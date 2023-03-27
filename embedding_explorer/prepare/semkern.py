"""Prepares semantic kernels for the given seeds
in a static word embedding model."""
from typing import List, NamedTuple, Tuple

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


def unique_connections(connections: np.ndarray) -> np.ndarray:
    # Making it so that in each edge the lower index comes first
    connections = np.stack(
        [np.min(connections, axis=1), np.max(connections, axis=1)]
    ).T
    # Then we run unique to get the unique pairs
    connections = np.unique(connections, axis=0)
    return connections


def get_associations(
    seed_ids: List[int],
    embeddings: np.ndarray,
    n_closest: int,
    metric: str = "cosine",
) -> np.ndarray:
    """Returns most closely related words to the given seeds in the form
    of edges from seeds to associations."""
    # Selecting terms
    selected_terms_matrix = embeddings[seed_ids]
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
    connections = []
    for i_seed, seed in enumerate(seed_ids):
        for association in closest[i_seed]:
            connections.append([seed, association])
    connections = np.array(connections)
    connections = unique_connections(connections)
    return connections


def calculate_dist_matrix(
    kernel_words: List[int], embeddings: np.ndarray
) -> np.ndarray:
    """Creates distance matrix of kernel words."""
    delta = pairwise_distances(embeddings[kernel_words])
    # Cut connections between the word and itself.
    np.fill_diagonal(delta, 0.0)
    # Cut connections that are over median distance
    # delta[delta < np.median(delta)] = 0.0
    return delta


class SemanticKernel(NamedTuple):
    vocabulary: np.ndarray  # array of str of shape (n_kernel, )
    connections: np.ndarray  # array of int of shape (n_connections, 2)
    priorities: np.ndarray  # array of {0, 1, 2} of shape (n_kernel)
    distance_matrix: np.ndarray  # array of float of shape (n_kernel, n_kernel)


def create_semantic_kernel(
    seed_ids: List[int],
    embeddings: np.ndarray,
    vocab: np.ndarray,
    n_first_level: int,
    n_second_level: int,
    metric: str = "cosine",
) -> SemanticKernel:
    # Collecting connections
    first_level_connections = get_associations(
        seed_ids, embeddings, n_closest=n_first_level, metric=metric
    )
    # Calculating which tokens come from the first level association
    first_level_tokens = list(
        set(first_level_connections.ravel()) - set(seed_ids)
    )
    second_level_connections = get_associations(
        first_level_tokens, embeddings, n_closest=n_second_level, metric=metric
    )
    # Calculating which tokens come from the second level association
    second_level_tokens = list(
        set(second_level_connections.ravel())
        - set(seed_ids)
        - set(first_level_connections.ravel())
    )
    # Removing the ones from second level that are associated with a seed
    # second_level_connections = second_level_connections[
    #     np.isin(second_level_connections, seed_ids).any(axis=1)
    # ]

    # Getting a set of unique tokens
    kernel_tokens = seed_ids + first_level_tokens + second_level_tokens
    kernel_priorities = np.array(
        [0] * len(seed_ids)
        + [1] * len(first_level_tokens)
        + [2] * len(second_level_tokens)
    )
    # Mapping ids of tokens to internal indices in the kernel
    # This is important for plotting
    id_to_index = {
        term_id: index for index, term_id in enumerate(kernel_tokens)
    }
    map_to_index = np.vectorize(id_to_index.get)
    first_level_connections = map_to_index(first_level_connections)
    second_level_connections = map_to_index(second_level_connections)
    # Collecting all connections
    connections = np.concatenate(
        [first_level_connections, second_level_connections], axis=0
    )
    # Collecting vocab
    kernel_vocab = vocab[kernel_tokens]
    # Calculating distance matrix
    distance_matrix = calculate_dist_matrix(
        kernel_tokens, embeddings=embeddings
    )
    return SemanticKernel(
        vocabulary=kernel_vocab,
        connections=connections,
        priorities=kernel_priorities,
        distance_matrix=distance_matrix,
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
    perplexity = min(distance_matrix.shape[0] - 1, 20)
    positions = TSNE(
        n_components=2,
        init="random",
        metric="precomputed",
        perplexity=perplexity,
    ).fit_transform(distance_matrix)
    x, y = positions.T
    return x, y


def calculate_n_connections(connections: np.ndarray) -> np.ndarray:
    """Calculates number of connections for each node in the graph."""
    n_kernel = connections.max() + 1
    n_connections = np.zeros(n_kernel)
    for connection in connections:
        for end_node in connection:
            n_connections[end_node] += 1
    return n_connections


def get_closest_seed(kernel: SemanticKernel) -> np.ndarray:
    """Returns order of closest seed to each token in the kernel."""
    seed_indices = kernel.priorities == 0
    distances_from_seeds = kernel.distance_matrix[:, seed_indices]
    closest_seed = np.argmin(distances_from_seeds, axis=1)
    return closest_seed
