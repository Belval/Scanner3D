"""
BaseGroupReg defines the basic interface to register more than 2 point clouds registration algorithms.
Returns a transformation and a fitness score.
"""

import abc


class BaseGroupReg(abc.ABC):
    def register(pcds):
        raise NotImplementedException
