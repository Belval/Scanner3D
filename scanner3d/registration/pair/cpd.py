"""
Uses CoeherentPointDrift to match two point clouds
"""

from probreg import cpd
from scanner3d.registration.pair.base_pair_reg import BasePairReg


class CPD(BasePairReg):
    def register(self, pcd1, pcd2):
        reg_p2p = cpd.registration_cpd(source, target)
        return reg_p2p.transformation, reg_p2p.fitness
