"""
Preprocessor that removes outliers from the point cloud
"""

from scanner3d.preprocess.base_preprocessor import BasePreprocessor

class RemoveOutliersPreprocessor(BasePreprocessor):
    def __init__(self, nb_neighbors=250, std_ratio=1.0):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def preprocess(self, pcd):
        return pcd.select_by_index(target.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1])
