"""
The scanner class defines the abstract class used by live and steps
"""

import abc
import open3d as o3d
import numpy as np
import seaborn as sns
from scanner3d.model import Model
from scanner3d.camera import Camera
from scanner3d.utils import cluster_hdbscan, points_to_plane


class Scanner(abc.ABC):
    @abc.abstractmethod
    def __init__(self, log_level):
        self.log_level = log_level
        self.pcd = None
        self.mesh = None
        self.camera = Camera(log_level=self.log_level)

    @abc.abstractmethod
    def start(self):
        pass

    def save_point_cloud(self, path: str = "out.pcd"):
        o3d.io.write_point_cloud(path, self.pcd)

    def load_point_cloud(self, path):
        self.pcd = o3d.io.read_point_cloud(path)

    def filter(self):
        # Keep every 10th point
        self.pcd = self.pcd.uniform_down_sample(10)
        # Empirically chosen values, should be changed
        self.pcd, _ = self.pcd.remove_statistical_outlier(
            nb_neighbors=200, std_ratio=3.0
        )

    def remove_support(self):
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        out = cluster_hdbscan(self.pcd.normals)

        max_score = -1
        remove_index = -1
        for i, v in enumerate(list(set(out))):
            res = points_to_plane(np.array(self.pcd.points)[out == v])
            if res > max_score:
                max_score = res
                remove_index = i

        color_count = len(list(set(out)))
        palette = sns.color_palette("hls", color_count)

        pcd_points = np.zeros((len(out), 3))
        pcd_colors = np.zeros((len(out), 3))
        for i, v in enumerate(out):
            if v == remove_index:
                continue
            pcd_points[i, :] = self.pcd.points[i]
            pcd_colors[i, :] = palette[v]

        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    def generate_mesh(
        self,
        reconstruction_method: str = "BPA",
        max_triangle_count: bool = 0,
        ensure_consistency: bool = True,
    ):
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        if reconstruction_method == "BPA":
            distances = self.pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                self.pcd, o3d.utility.DoubleVector([radius / 2, radius, radius * 2])
            )
        elif reconstruction_method == "Poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                self.pcd
            )
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            mesh = mesh.crop(self.pcd.get_axis_aligned_bounding_box())
        else:
            raise NotImplementedError()
        if max_triangle_count > 0:
            mesh = mesh.simplify_quadric_decimation(args.max_triangle_count)
        if ensure_consistency:
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
        self.mesh = mesh
        return mesh

    def smooth(self):
        return self.mesh.filter_smooth_laplacian()

    def model(self):
        if self.mesh:
            return Model(self.mesh)
        else:
            return Model(self.generate_mesh())
