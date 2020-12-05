"""
BasePairReg defines the basic interface used by all pair (2 point cloud) registration algorithms.
Returns a transformation and a fitness score.
"""

import abc

class BasePairReg(abc.ABC):
    def register(pcd1, pcd2):
        raise NotImplementedException
