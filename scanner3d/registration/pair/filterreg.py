"""
Uses FilterReg algorithm to match two point clouds
"""

import numpy as np
import open3d as o3d
from probreg import filterreg
from scanner3d.registration.pair.base_pair_reg import BasePairReg


class FilterReg(BasePairReg):
    def register(self, pcd1, pcd2):
        f_reg = filterreg.registration_filterreg(
            pcd1,
            pcd2,
            tol=0.00001,
            maxiter=500,
            objective_type="pt2pl",
            feature_fn=features.FPFH(),
        )
        return f_reg.q, f_reg.transformation
