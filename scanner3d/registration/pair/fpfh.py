"""
Uses FeatureMatching to match two point clouds
"""

import numpy as np
import open3d as o3d
from probreg import svr
from scanner3d.registration.pair.base_pair_reg import BasePairReg

class SVR(BasePairReg):
    def register(self, pcd1, pcd2):
        reg_p2p = o3d.registration.registration_fast_based_on_feature_matching(pcd1, pcd2)
        return reg_p2p.transformation, reg_p2p.fitness

o3d.registration.registration_fast_based_on_feature_matching(