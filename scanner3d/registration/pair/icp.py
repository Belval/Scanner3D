"""
Uses FilterReg algorithm to match two point clouds
"""

import numpy as np
import open3d as o3d
from probreg import filterreg
from scanner3d.registration.pair.base_pair_reg import BasePairReg


class ICP(BasePairReg):
    def register(self, pcd1, pcd2):
        reg_p2p = o3d.registration.registration_icp(
            pcd1,
            pcd2,
            0.1,
            np.identity(4),
            o3d.registration.TransformationEstimationPointToPlane(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        return reg_p2p.transformation, reg_p2p.fitness
