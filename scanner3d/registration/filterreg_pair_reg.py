"""
Uses FilterReg algorithm to match two point clouds
"""

import numpy as np
import open3d as o3d
from probreg import filterreg
from scanner3d.registration.base_pair_reg import BasePairReg


class FilterReg(BasePairReg):
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
        threshold = 0.01
        reg_p2p = o3d.registration.registration_icp(
            source,
            target,
            0.1,
            np.identity(4),
            o3d.registration.TransformationEstimationPointToPlane(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        return reg_p2p.transformation, reg_p2p.fitness
