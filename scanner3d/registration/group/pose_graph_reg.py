"""
Open3D implements multiway registration via pose graph optimization.
The backend implements the technique presented in [Choi2015].
"""

import logging
import numpy as np
import open3d as o3d

from scanner3d.registration.base_group_reg import BaseGroupReg


class PoseGraphReg(BaseGroupReg):
    def __init__(self, voxel_size):
        # FIXME: Allow user to define a pair reg algorithm
        self.voxel_size = voxel_size
        self.max_correspondence_distance_coarse = self.voxel_size * 15
        self.max_correspondence_distance_fine = self.voxel_size * 1.5

    def pairwise_registration(self, source, target):
        logging.info("Apply point-to-plane ICP")
        icp_coarse = o3d.registration.registration_icp(
            source,
            target,
            self.max_correspondence_distance_coarse,
            np.identity(4),
            o3d.registration.TransformationEstimationPointToPlane(),
        )
        icp_fine = o3d.registration.registration_icp(
            source,
            target,
            self.max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.registration.TransformationEstimationPointToPlane(),
        )
        transformation_icp = icp_fine.transformation
        information_icp = o3d.registration.get_information_matrix_from_point_clouds(
            source,
            target,
            self.max_correspondence_distance_fine,
            icp_fine.transformation,
        )
        return transformation_icp, information_icp

    def full_registration(
        self, pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine
    ):
        pose_graph = o3d.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = self.pairwise_registration(
                    pcds[source_id], pcds[target_id]
                )
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.registration.PoseGraphNode(np.linalg.inv(odometry))
                    )
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(
                            source_id,
                            target_id,
                            transformation_icp,
                            information_icp,
                            uncertain=False,
                        )
                    )
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(
                            source_id,
                            target_id,
                            transformation_icp,
                            information_icp,
                            uncertain=True,
                        )
                    )
        return pose_graph

    def register(self, pcds, pair_reg=None):
        pose_graph = self.full_registration(
            pcds,
            self.max_correspondence_distance_coarse,
            self.max_correspondence_distance_fine,
        )

        logging.info("Optimizing PoseGraph ...")
        option = o3d.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.registration.global_optimization(
            pose_graph,
            o3d.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        logging.info("Transform points and display")
        for point_id in range(len(pcds)):
            pcds[point_id].transform(pose_graph.nodes[point_id].pose)

        logging.info("Returning transforms")
        transforms = []
        for point_id in range(len(pcds)):
            transforms.append(pose_graph.nodes[point_id].pose)

        return transforms
