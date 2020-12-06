"""
Open3D implements multiway registration via pose graph optimization.
The backend implements the technique presented in [Choi2015].
"""

import logging
import numpy as np
import open3d as o3d

from scanner3d.registration.group.base_group_reg import BaseGroupReg


class PoseGraphReg(BaseGroupReg):
    def __init__(self, voxel_size, pair_reg=None):
        # FIXME: Allow user to define a pair reg algorithm
        self.voxel_size = voxel_size
        self.max_distance_coarse = self.voxel_size * 15
        self.max_distance_fine = self.voxel_size * 1.5

    def pair_reg(self, pcd1, pcd2):
        icp_coarse = o3d.registration.registration_icp(
            pcd1,
            pcd2,
            self.max_distance_coarse,
            np.identity(4),
            o3d.registration.TransformationEstimationPointToPlane(),
        )
        icp_fine = o3d.registration.registration_icp(
            pcd1,
            pcd2,
            self.max_distance_fine,
            icp_coarse.transformation,
            o3d.registration.TransformationEstimationPointToPlane(),
        )
        transformation_icp = icp_fine.transformation
        information_icp = o3d.registration.get_information_matrix_from_point_clouds(
            pcd1,
            pcd2,
            self.max_distance_fine,
            icp_fine.transformation,
        )
        return transformation_icp, information_icp

    def register(self, pcds, pair_reg=None):
        pose_graph = o3d.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
        for i in range(len(pcds)):
            for j in range(i + 1, len(pcds)):
                trans, info = self.pair_reg(pcds[i], pcds[j])
                if j == i + 1:
                    odometry = np.dot(trans, odometry)
                    pose_graph.nodes.append(
                        o3d.registration.PoseGraphNode(np.linalg.inv(odometry))
                    )
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(
                            i, j, trans, info, uncertain=False
                        )
                    )
                else:
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(
                            i, j, trans, info, uncertain=True
                        )
                    )

        option = o3d.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.registration.global_optimization(
            pose_graph,
            o3d.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        for i in range(len(pcds)):
            pcds[i].transform(pose_graph.nodes[i].pose)

        transforms = []
        for i in range(len(pcds)):
            transforms.append(pose_graph.nodes[i].pose)

        return transforms
