"""
Model object, can be saved to disk
"""

import open3d as o3d


class Model:
    def __init__(self, mesh):
        self.mesh = mesh

    def save(self, path):
        o3d.io.write_triangle_mesh(path, self.mesh)
