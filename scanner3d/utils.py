import hdbscan
import numpy as np
from sklearn.metrics import pairwise_distances


def cluster_hdbscan(data):
    distance = pairwise_distances(data, metric="cosine")
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed", min_cluster_size=int(len(data) * 0.1)
    )
    clusterer.fit(distance.astype("float64"))
    return clusterer.labels_


def points_to_plane(points):
    normalized_points = np.transpose(
        np.transpose(points) - np.sum(points, 1) / len(points)
    )
    svd = np.transpose(np.linalg.svd(normalized_points))
    return abs(np.sum(np.dot(points, np.transpose(svd[1]))) / len(points))
