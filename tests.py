"""
Unit testing
"""

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


class TestPairRegistration(unittest.TestCase):
    def test_custom_icp(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        trans, fit = CustomICP().register(pcd1, pcd2)
        self.assertTrue(fit > 0.97)

    def test_cpd(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        trans, fit = CPD().register(pcd1, pcd2)
        self.assertTrue(fit > 0.97)

    def test_filterreg(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        trans, fit = FilterReg().register(pcd1, pcd2)
        self.assertTrue(fit > 0.97)

    def test_fpfh(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        trans, fit = FPFH().register(pcd1, pcd2)
        self.assertTrue(fit > 0.97)

    def test_icp(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        trans, fit = ICP().register(pcd1, pcd2)
        self.assertTrue(fit > 0.97)

    def test_svr(self):
        pcd1 = o3d.io.read_point_cloud("tests/artefacts/bird_1.pcd")
        pcd2 = o3d.io.read_point_cloud("tests/artefacts/bird_2.pcd")
        trans, fit = SVR().register(pcd1, pcd2)
        self.assertTrue(fit > 0.97)

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
