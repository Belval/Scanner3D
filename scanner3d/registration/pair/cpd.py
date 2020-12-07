"""
Uses Coeherent Point Drift to match two point clouds
"""

from probreg import cpd
from scanner3d.registration.pair.base_pair_reg import BasePairReg


class CPD(BasePairReg):
    def register(self, pcd1, pcd2):
        reg = cpd.registration_cpd(pcd1, pcd2)
        return reg.transformation
