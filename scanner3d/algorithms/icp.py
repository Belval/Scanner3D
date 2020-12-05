"""
This is a manual implementation of (Point-to-point) ICP, it is not as optimized
as Open3D's nor does as fully-featured.

It's also quite slow.
"""

import numpy as np
from scipy.spatial.distance import cdist

from scanner3d.exceptions import PointCloudSizeMismatch


def best_fit_transform(source, dest):
    if len(source) != len(dest):
        raise

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def nearest_neighbor(src, dst):
    all_dists = cdist(src, dst, "euclidean")
    indices = all_dists.argmin(axis=1)
    distances = all_dists[np.arange(all_dists.shape[0]), indices]
    return distances, indices


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[0:3, :] = np.copy(A.T)
    dst[0:3, :] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[0:3, :].T, dst[0:3, :].T)

        T, _, _ = best_fit_transform(src[0:3, :].T, dst[0:3, indices].T)

        src = np.dot(T, src)

        mean_error = np.sum(distances) / distances.size
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T, _, _ = best_fit_transform(A, src[0:3, :].T)

    return T
