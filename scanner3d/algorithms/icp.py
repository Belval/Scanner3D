"""
This is a manual implementation of (Point-to-point) ICP, it is not as optimized
as Open3D's nor does as fully-featured.

It's also quite slow.
"""

import numpy as np
from scipy.spatial.distance import cdist

from scanner3d.exceptions import PointCloudSizeMismatch


def find_transform(pcd1, pcd2):
    center_pcd1 = np.mean(pcd1, axis=0)
    center_pcd2 = np.mean(pcd2, axis=0)
    centered_pcd_1 = pcd1 - center_pcd1
    centered_pcd_2 = pcd2 - center_pcd2

    U, S, Vt = np.linalg.svd(np.dot(centered_pcd_1.T, centered_pcd_2))
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = center_pcd2.T - np.dot(R, center_pcd1.T)

    transformation_matrix = np.identity(4)
    transformation_matrix[0:3, 0:3] = R
    transformation_matrix[0:3, 3] = t

    return transformation_matrix


def find_nn(src, dst):
    dists = cdist(src, dst, "euclidean")
    indices = dists.argmin(axis=1)
    distances = dists[np.arange(dists.shape[0]), indices]
    return distances, indices


def icp(pcd1, pcd2, max_iterations=20, tolerance=0.001):
    source = np.ones((4, np.array(pcd1.points).shape[0]))
    dest = np.ones((4, np.array(pcd2.points).shape[0]))
    source[0:3, :] = np.copy(np.array(pcd1.points).T)
    dest[0:3, :] = np.copy(np.array(pcd2.points).T)

    prev_error = None

    for i in range(max_iterations):
        distances, indices = find_nn(source[0:3, :].T, dest[0:3, :].T)

        transformation_matrix = find_transform(source[0:3, :].T, dest[0:3, indices].T)

        source = np.dot(transformation_matrix, source)

        mean_error = np.sum(distances) / distances.size

        if prev_error is None or abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    transformation_matrix = find_transform(source[0:3, :].T, dest[0:3, indices].T)

    return transformation_matrix
