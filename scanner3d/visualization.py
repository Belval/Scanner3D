import logging
import numpy as np
import open3d as o3d
import time
import queue
import logging
from multiprocessing import Process, Manager

class Visualization:
    def __init__(self, pcd, log_level):
        self.log_level = log_level
        self.pcd = pcd
        self.visualizer = Process(
            target=Visualization.show_window,
            args=(
                np.asarray(self.pcd.points),
                np.asarray(self.pcd.colors),
                np.asarray(self.pcd.normals)
            )
        )
        self.visualizer.start()

    def update(self, pcd):
        self.pcd = pcd
        self.visualizer.kill()
        self.visualizer = Process(
            target=Visualization.show_window,
            args=(
                np.asarray(self.pcd.points),
                np.asarray(self.pcd.colors),
                np.asarray(self.pcd.normals)
            )
        )
        self.visualizer.start()

    def stop(self):
        self.visualizer.kill()

    def show_window(p, c, n):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        pcd.colors = o3d.utility.Vector3dVector(c)
        pcd.normals = o3d.utility.Vector3dVector(n)

        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(5.0, 0.0)
            return False

        o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)
