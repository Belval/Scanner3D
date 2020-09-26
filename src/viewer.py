import os
import copy
import argparse
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from functools import partial

def main(args):
    # Reset all devices
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()

    time.sleep(5)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intrinsics = depth_profile.get_intrinsics()

    # Point cloud acquisition
    pc = rs.pointcloud()
    colorizer = rs.colorizer()
    aligner = rs.align(rs.stream.color)

    geometry_added = False
    pcd = None

    def save_pcd(depth_image, color_image):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            convert_rgb_to_intensity=False
        )
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.io.write_point_cloud(f"clouds/{time.time()}.pcd", pcd)

    while True:
        frames = aligner.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow("3D Scanner", images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Saving point cloud")
            save_pcd(depth_image, color_image)
    
    # ICP point cloud joining
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])
    
    def pairwise_registration(source, target):
        print("Apply point-to-plane ICP")
        icp_coarse = o3d.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.registration.registration_icp(
            source, target, max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation
        information_icp = o3d.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp


    def full_registration(pcds, max_correspondence_distance_coarse,
                        max_correspondence_distance_fine):
        pose_graph = o3d.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = pairwise_registration(
                    pcds[source_id], pcds[target_id])
                print("Build o3d.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=True))
        return pose_graph

    pcds = [o3d.io.read_point_cloud(os.path.join("clouds", f)) for f in os.listdir("clouds")]
    target = pcds[0]
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    for pcd in pcds[1:]:
        source = pcd
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        threshold = 0.1
        reg_p2p = o3d.registration.registration_icp(
            source, target, threshold, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            o3d.registration.TransformationEstimationPointToPlane(),
            o3d.registration.ICPConvergenceCriteria(max_iteration = 20000000)
        )
        print(reg_p2p)
        draw_registration_result(source, target, reg_p2p.transformation)
        if reg_p2p.fitness > 0.75:
            source.transform(reg_p2p.transformation)
            target = source + target
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pcd = target
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if args.mesh_reconstruction_method == "BPA":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
        if args.max_triangle_count > 0:
            mesh = mesh.simplify_quadric_decimation(args.max_triangle_count)
        if args.ensure_consistency:
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
    elif args.mesh_reconstruction_method == "Poisson":
        raise NotImplementedError()
    else: 
        raise NotImplementedError()

    o3d.visualization.draw_geometries([mesh])

    o3d.io.write_triangle_mesh(args.output_file, mesh)

    pipeline.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scan a 3D object and save the mesh file")
    parser.add_argument(
        "--output-file", type=str, nargs="?", help="Name of the output file", default="result.obj"
    )
    parser.add_argument(
        "--mesh-reconstruction-method", type=str, nargs="?", help="Method that will be used to create a mesh from the point cloud. One of [BPA|Poisson]", default="BPA"
    )
    parser.add_argument(
        "--max-triangle-count", type=int, nargs="?", help="Maximum triangle count, default is -1 (None)", default=-1
    )
    parser.add_argument(
        "--ensure-consistency", action="store_true", help="Whether to clean the resulting mesh or not"
    )
    main(parser.parse_args())
