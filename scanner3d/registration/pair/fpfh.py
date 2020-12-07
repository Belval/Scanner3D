"""
Uses FeatureMatching to match two point clouds
"""

import numpy as np
import open3d as o3d
from scanner3d.registration.pair.base_pair_reg import BasePairReg


class FPFH(BasePairReg):
    def register(self, pcd1, pcd2):
        reg_fpfh = o3d.registration.registration_fast_based_on_feature_matching(
            pcd1,
            pcd2,
            o3d.registration.compute_fpfh_feature(pcd1, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)),
            o3d.registration.compute_fpfh_feature(pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
        )
        return reg_fpfh.transformation
