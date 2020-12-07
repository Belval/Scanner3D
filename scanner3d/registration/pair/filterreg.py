"""
Uses FilterReg algorithm to match two point clouds
"""

import numpy as np
import open3d as o3d
from probreg import filterreg
from scanner3d.registration.pair.base_pair_reg import BasePairReg


class FilterReg(BasePairReg):
    def register(self, pcd1, pcd2):
        f_reg = filterreg.registration_filterreg(pcd1, pcd2)
        trans = f_reg.transformation
        scale_matrix = np.identity(4) * trans.scale
        transformation_matrix = np.identity(4)
        transformation_matrix[0:3, 0:3] = trans.rot
        transformation_matrix[0:3, 3] = trans.t
        transformation_matrix *= trans.scale
        return transformation_matrix