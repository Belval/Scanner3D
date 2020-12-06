"""
Use the custom ICP algorithm implemented in algorithms/icp.py to match two point clouds
"""

from scanner3d.registration.pair.base_pair_reg import BasePairReg
from scanner3d.algorithms.icp import icp


class CustomICP(BasePairReg):
    def register(self, pcd1, pcd2):
        trans, fit = icp(pcd1, pcd2)
        return trans, fit
