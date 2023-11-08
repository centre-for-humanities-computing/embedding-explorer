import json
import warnings
from typing import Optional, Tuple

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

REDUCTION_METHODS = {
    "umap": UMAP,
    "svd": TruncatedSVD,
    "nmf": NMF,
}


def get_reduced_embeddings(
    embeddings: np.ndarray,
    do_dim_red: bool,
    dim_red_method: str,
    n_dimensions: int,
    dim_red_params: str,
) -> np.ndarray:
    if not do_dim_red:
        return embeddings
    try:
        kwargs = json.loads(dim_red_params)
    except json.JSONDecodeError:
        warnings.warn("Couldn't dimensionality reduction parameters as JSON.")
        kwargs = dict()
    model = REDUCTION_METHODS[dim_red_method](
        n_components=n_dimensions, **kwargs
    )
    return model.fit_transform(embeddings)


CLUSTERING_METHODS = {
    "kmeans": KMeans,
    "hdbscan": HDBSCAN,
    "spectral": SpectralClustering,
}


def get_clustering(
    reduced_embeddings: np.ndarray,
    do_cluster: bool,
    clustering_method: str,
    n_clusters: int,
    clustering_params: str,
) -> Optional[np.ndarray]:
    if not do_cluster:
        return None
    if clustering_method == "hdbscan":
        args = dict()
    else:
        args = dict(n_clusters=n_clusters)
    try:
        kwargs = json.loads(clustering_params)
    except json.JSONDecodeError:
        warnings.warn("Couldn't parse clustering parameters as JSON.")
        kwargs = dict()
    model = CLUSTERING_METHODS[clustering_method](**args, **kwargs)
    model.fit(reduced_embeddings)
    return model.labels_


PROJECTION_METHODS = {
    "umap": UMAP,
    "svd": TruncatedSVD,
    "tsne": TSNE,
}


def get_projection(
    reduced_embeddings: np.ndarray,
    projection_method: str,
    projection_params: str,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        kwargs = json.loads(projection_params)
    except json.JSONDecodeError:
        warnings.warn("Couldn't parse projection parameters as JSON.")
        kwargs = dict()
    model = PROJECTION_METHODS[projection_method](n_components=2, **kwargs)
    x, y = model.fit_transform(reduced_embeddings).T
    return x, y
