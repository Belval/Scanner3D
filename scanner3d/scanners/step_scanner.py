"""
steps runs the 3D reconstruction process in steps.

1- Acquisition
2- Registration

Very similar to live_scanner, but takes a group registration algorithm instead of
a pair registration algorithm.
"""

import cv2
import logging
import numpy as np
import open3d as o3d

import os
import time

from scanner3d.registration.group.base_group_reg import BaseGroupReg
from scanner3d.scanners.scanner import Scanner
from scanner3d.camera import Camera


class StepScanner(Scanner):
    def __init__(
        self, log_level, registration_algorithm: BaseGroupReg, cloud_dir: str = None
    ):
        super(StepScanner, self).__init__(log_level)
        # self.camera = Camera(log_level)
        self.reg = registration_algorithm
        self.vis = None
        self.pcd = None
        self.continuous_capture = False
        self.rotated_capture = False
        self.pcds = (
            []
            if cloud_dir is None
            else [
                o3d.io.read_point_cloud(os.path.join(cloud_dir, f))
                for f in os.listdir(cloud_dir)
            ]
        )
        self.trans_matrices = []

    def start(self):
        logging.info("Starting acquisition in step scanner")

        window = cv2.namedWindow("3D Scanner", cv2.WINDOW_NORMAL)
        self.continuous_capture = False
        self.rotated_capture = False
        while cv2.getWindowProperty("3D Scanner", cv2.WND_PROP_VISIBLE) >= 1:
            color_image, depth_colormap = self.camera.image_depth()
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow("3D Scanner", images)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                self.camera.stop()
                break

            if cv2.waitKey(1) & 0xFF == ord("r"):
                print("Rotated capture toggled")

            if cv2.waitKey(1) & 0xFF == ord("c"):
                self.pcds.pop()
                self.vis.update(self.pcd)
                continue

            if cv2.waitKey(1) & 0xFF == ord("g"):
                print("Continuous capture toggled")
                self.continuous_capture = not self.continuous_capture

            if cv2.waitKey(1) & 0xFF == ord("s"):
                print("Saving point cloud")
                pcd = self.camera.pcd()
                o3d.io.write_point_cloud(f"clouds/{time.time()}.pcd", pcd)

            if self.continuous_capture:
                pcd = self.camera.pcd()
                o3d.io.write_point_cloud(f"clouds/{time.time()}.pcd", pcd)

        self.camera.stop()
        cv2.destroyAllWindows()
        return
        logging.info("Starting registration in step scanner")
        pcds = [o3d.io.read_point_cloud("clouds/" + f) for f in os.listdir("clouds/")]
        for pcd in pcds:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        downsampled_pcds = [pcd.voxel_down_sample(voxel_size=0.01) for pcd in pcds]
        transformations = self.reg.register(downsampled_pcds)

        pcd_combined = o3d.geometry.PointCloud()
        for pcd, trans in zip(pcds, transformations):
            pcd.transform(trans)
            pcd_combined += pcd

        self.pcd = pcd_combined
        o3d.visualization.draw_geometries([self.pcd])
        self.save_point_cloud()
