"""
Fitness is a value between 0 and 1 where 1 is a perfect alignment (given a threshold)
"""

import numpy as np
from sklearn.neighbors import KDTree
from .base_metric import BaseMetric


class RMSE(BaseMetric):
    def compute(self, pcd1, pcd2):
        # pcd1 is source, pcd2 is target, score of 1 means all points
        # of pcd1 are within $threshold or a point of pcd2
        kdt = KDTree(np.array(pcd2.points), leaf_size=30, metric="euclidean")
        ds, _ = kdt.query(np.array(pcd1.points), k=1)
        return np.sqrt(np.sum(ds * ds) / len(ds))
