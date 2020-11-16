"""
live runs the registration process during the acquisition.
This allows the user to get a general idea of what the end model will look like.
"""

import copy
import cv2
import numpy as np
import logging

from scanner3d.camera import Camera
from scanner3d.scanners.scanner import Scanner
from scanner3d.registration.base_pair_reg import BasePairReg
from scanner3d.visualization import Visualization


class LiveScanner(Scanner):
    def __init__(
        self, log_level, registration_algorithm: BasePairReg, cloud_dir: str = None
    ):
        super(LiveScanner, self).__init__(log_level)
        self.camera = Camera(log_level)
        self.reg = registration_algorithm
        self.vis = None
        self.pcd = None
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
        window = cv2.namedWindow("3D Scanner", cv2.WINDOW_NORMAL)
        while cv2.getWindowProperty("3D Scanner", cv2.WND_PROP_VISIBLE) >= 1:
            color_image, depth_colormap = self.camera.image_depth()
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow("3D Scanner", images)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                camera.stop()
                break

            if cv2.waitKey(1) & 0xFF == ord("c"):
                self.pcds.pop()
                self.vis.update(self.pcd)
                continue

            if cv2.waitKey(1) & 0xFF == ord("s"):
                print("Saving point cloud")
                pcd = self.camera.pcd()
                if not self.pcds:
                    self.pcds.append(pcd)
                    self.pcd = pcd
                    self.vis = Visualization(self.pcd, self.log_level)
                    continue
                trans_matrix, fit = self.reg.register(self.pcds[-1], pcd)
                if fit > 0.97:
                    logging.info(f"Cloud matched ({fit}), merging...")
                    self.trans_matrices.append(trans_matrix)
                    self.pcds.append(pcd)
                    source = copy.deepcopy(self.pcds[-1])
                    if isinstance(trans_matrix, np.ndarray):
                        source.transform(trans_matrix)
                    else:
                        source.points = trans_matrix.transform(source.points)
                    self.pcd = self.pcd + source
                    self.vis.update(self.pcd)
                else:
                    logging.warning("Chain broken, go back.")

        self.camera.stop()
        if self.vis is not None:
            self.vis.stop()
        cv2.destroyAllWindows()
