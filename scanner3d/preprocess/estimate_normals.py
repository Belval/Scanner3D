"""
Preprocessor estimates normals on a point cloud
"""

import open3d as o3d
from scanner3d.preprocess.base_preprocessor import BasePreprocessor


class EstimateNormalsPreprocessor(BasePreprocessor):
    def __init__(self, radius=0.1, max_nn=30):
        self.radius = radius
        self.max_nn = max_nn

    def preprocess(self, pcd):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.radius, max_nn=self.max_nn
            )
        )
        return pcd
