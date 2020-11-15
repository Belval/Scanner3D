"""
Model object, can be saved to disk
"""


class Model:
    def __init__(self, mesh):
        self.mesh = mesh

    def save(self, path):
        o3d.io.write_triangle_mesh(path, mesh)
