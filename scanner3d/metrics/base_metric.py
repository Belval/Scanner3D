"""
A metric is a class that takes two point cloud and returns a value that
defines how tight the alignment is.
"""

import abc


class BaseMetric(abc.ABC):
    def compute(self, pcd1, pcd2):
        raise NotImplementedException
