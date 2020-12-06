"""
Preprocessor uses the voxel technique to downsample a point cloud
"""

from scanner3d.preprocess.base_preprocessor import BasePreprocessor


class DownsamplePreprocessor(BasePreprocessor):
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size

    def preprocess(self, pcd):
        return pcd.voxel_down_sample(voxel_size=self.voxel_size)
