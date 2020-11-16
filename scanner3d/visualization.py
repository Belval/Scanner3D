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
        self.manager = Manager()
        self.queue = self.manager.Queue()
        self.visualizer = Process(
            target=Visualization.loop,
            args=(self.queue, self.log_level)
        )
        self.queue.put((
            np.asarray(pcd.points),
            np.asarray(pcd.colors),
            np.asarray(pcd.normals)
        ))
        self.visualizer.start()

    def update(self, pcd):
        self.queue.put((
            np.asarray(pcd.points),
            np.asarray(pcd.colors),
            np.asarray(pcd.normals)
        ))

    def loop(queue, log_level):
        logging.basicConfig(level=log_level)
        logging.debug(f"Visualization loop started")
        p, c, n = queue.get()
        logging.debug(f"Got initial point cloud from queue")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        logging.debug(f"Instantiated point cloud {pcd}")
        pcd.colors = o3d.utility.Vector3dVector(c)
        pcd.normals = o3d.utility.Vector3dVector(n)

        logging.debug(f"Visualizer started")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        logging.debug(f"Showing window")
        vis.run()
        logging.debug(f"Window should be visible")
        
        while True:
            logging.debug(f"Looping")
            try:
                p, c, n = queue.get(block=False)
                pcd.points = o3d.utility.Vector3dVector(p)
                pcd.colors = o3d.utility.Vector3dVector(c)
                pcd.normals = o3d.utility.Vector3dVector(n)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            except Exception as ex:
                logging.info(ex)
                time.sleep(1)
