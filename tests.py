"""
Unit testing
"""

import time
import unittest
import open3d as o3d

from scanner3d.registration.pair import (
    CPD,
    CustomICP,
    FilterReg,
    FPFH,
    ICP,
    SVR,
)
from scanner3d.registration.group import (
    PoseGraph,
    BestMatch,
)
from scanner3d.preprocess import (
    DownsamplePreprocessor,
    EstimateNormalsPreprocessor,
    PreprocessorSequence,
    RemoveOutliersPreprocessor,
)
from scanner3d.metrics import Fitness, RMSE

base_preprocessor = PreprocessorSequence([
    EstimateNormalsPreprocessor(radius=0.1, max_nn=30),
    # For tests we downsample so limit runtime
    DownsamplePreprocessor(voxel_size=0.01),
])

class TestPairRegistration(unittest.TestCase):
    def test_custom_icp(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        pcd1, pcd2 = base_preprocessor.preprocess([pcd1, pcd2])
        trans = CustomICP().register(pcd1, pcd2)
        pcd1.transform(trans)
        self.assertTrue(Fitness().compute(pcd1, pcd2) > 0.81)
        self.assertTrue(RMSE().compute(pcd1, pcd2) < 0.04)

    # Takes way too long to be a test
    #def test_cpd(self):
    #    pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
    #    pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
    #    pcd1, pcd2 = base_preprocessor.preprocess([pcd1, pcd2])
    #    trans, fit = CPD().register(pcd1, pcd2)
    #    self.assertTrue(fit > 0.97)

    def test_filterreg(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        pcd1, pcd2 = base_preprocessor.preprocess([pcd1, pcd2])
        start = time.time()
        trans = FilterReg().register(pcd1, pcd2)
        pcd1.transform(trans)
        self.assertTrue(Fitness().compute(pcd1, pcd2) > 0.97)
        self.assertTrue(RMSE().compute(pcd1, pcd2) < 0.04)

    def test_fpfh(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        pcd1, pcd2 = base_preprocessor.preprocess([pcd1, pcd2])
        start = time.time()
        trans = FPFH().register(pcd1, pcd2)
        pcd1.transform(trans)
        self.assertTrue(Fitness().compute(pcd1, pcd2) > 0.93)
        self.assertTrue(RMSE().compute(pcd1, pcd2) < 0.03)

    def test_icp(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        pcd1, pcd2 = base_preprocessor.preprocess([pcd1, pcd2])
        start = time.time()
        trans = ICP().register(pcd1, pcd2)
        pcd1.transform(trans)
        self.assertTrue(Fitness().compute(pcd1, pcd2) > 0.86)
        self.assertTrue(RMSE().compute(pcd1, pcd2) < 0.04)

    def test_svr(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        pcd1, pcd2 = base_preprocessor.preprocess([pcd1, pcd2])
        start = time.time()
        trans = SVR().register(pcd1, pcd2)
        pcd1.transform(trans)
        self.assertTrue(Fitness().compute(pcd1, pcd2) > 0.97)
        self.assertTrue(RMSE().compute(pcd1, pcd2) < 0.97)

class TestGroupRegistration(unittest.TestCase):
    def test_pose_graph(self):
        pass

    def test_best_match(self):
        pass


class TestPreprocessor(unittest.TestCase):
    def test_downsample(self):
        pass

    def test_estimate_normals(self):
        pass

    def test_remove_outliers(self):
        pass

    def test_preprocessor_sequence(self):
        pass


if __name__ == "__main__":
    unittest.main()
