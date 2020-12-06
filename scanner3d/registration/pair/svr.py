"""
Uses Support Vector Registration to match two point clouds
"""

import numpy as np
import open3d as o3d
from probreg import l2dist_regs
from scanner3d.registration.pair.base_pair_reg import BasePairReg


class SVR(BasePairReg):
    def register(self, pcd1, pcd2):
        reg_p2p = l2dist_regs.registration_svr(pcd1, pcd2)
        return reg_p2p.transformation, reg_p2p.fitness
