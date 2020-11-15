"""
The scanner class defines the abstract class used by live and steps
"""

import abc
from scanner3d.model import Model
from scanner3d.camera import Camera


class Scanner(abc.ABC):
    @abc.abstractmethod
    def __init__(self, log_level):
        self.camera = Camera(log_level=log_level)

    @abc.abstractmethod
    def start(self):
        pass

    def mesh(
        self,
        reconstruction_method: str = "BPA",
        max_triangle_count: bool = 0,
        ensure_consistency: bool = True,
    ):
        if mesh_reconstruction_method == "BPA":
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([radius, radius * 2])
            )
            if max_triangle_count > 0:
                mesh = mesh.simplify_quadric_decimation(args.max_triangle_count)
            if ensure_consistency:
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
        elif args.mesh_reconstruction_method == "Poisson":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return mesh

    def model(self):
        return Model(self.mesh())
