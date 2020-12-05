"""
Use the custom ICP algorithm implemented in algorithms/icp.py to match two point clouds
"""

import numpy as np
import open3d as o3d
from probreg import filterreg
from scanner3d.registration.base_pair_reg import BasePairReg
from scanner3d.algorithms.icp import icp


class CustomICP(BasePairReg):
    def register(self, pcd1, pcd2):
        target = pcd1
        target = target.select_by_index(
            target.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1]
        )
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        source = pcd2
        source = source.select_by_index(
            source.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1]
        )
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        trans, fit = icp(source, dest)

        return trans, fit
