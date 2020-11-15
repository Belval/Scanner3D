import logging
import open3d as o3d
from multiprocessing import Process, Manager

class Visualization:
    def __init__(self, pcd):
        self.pcd = pcd
        self.start()

    def start(self):
        o3d.io.write_point_cloud(f"temp.pcd", self.pcd)
        self.vis_process = Process(target=Visualization.visualize, args=("temp.pcd",))
        self.vis_process.start()

    def update(self, pcd):
        self.pcd = pcd
        self.vis_process.kill()
        self.start()

    def visualize(pcd):
        logging.info("Updating visualization")
        o3d.visualization.draw_geometries([o3d.io.read_point_cloud("temp.pcd")])
