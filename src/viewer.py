import os
import copy
import argparse
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from probreg import filterreg
from tqdm import tqdm
from functools import partial

def save_pcd(depth_image, color_image):
    pcd = create_pcd(depth_image, color_image)
    o3d.io.write_point_cloud(f"clouds/{time.time()}.pcd", pcd)

def create_pcd(depth_image, color_image, intrinsics):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        convert_rgb_to_intensity=False
    )
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def build_from_dir(path):
    pcds = [o3d.io.read_point_cloud(os.path.join(path, f)) for f in os.listdir(path)]
    
    trans_matrices = []
    with tqdm(total=len(pcds) - 1) as t:
        for i, pcd in enumerate(pcds[1:]):
            target = pcds[i]
            target = target.select_by_index(target.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1])
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            source = pcd
            source = source.select_by_index(source.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1])
            source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            threshold = 0.1
            reg_p2p = o3d.registration.registration_colored_icp(
                source, target, threshold, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                #o3d.registration.TransformationEstimationPointToPlane(),
                o3d.registration.ICPConvergenceCriteria(max_iteration = 2000)
            )
            if reg_p2p.fitness > 0.75:
                trans_matrices.append(reg_p2p.transformation)
            else:
                raise Exception("Chain broken")
            t.update(1)
    
    merge = pcds[0]
    for i, pcd in enumerate(pcds[1:]):
        for m in trans_matrices[:i+1]:
            pcd.transform(m)
        merge = merge + pcd.select_by_index(pcd.remove_statistical_outlier(nb_neighbors=250, std_ratio=0.1)[1])
    merge = merge.select_by_index(merge.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1])
    return merge

def get_transform(pcd1, pcd2):
    target = pcd1
    target = target.select_by_index(target.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1])
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    source = pcd2
    source = source.select_by_index(source.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1])
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    threshold = 0.1
    reg_p2p = filterreg.registration_filterreg(source, target)
    print(reg_p2p.fitness)
    if reg_p2p.fitness > 0.98:
        return reg_p2p.transformation
    else:
        raise Exception("Chain broken")

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
    transforms = []
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
            pcd = create_pcd(depth_image, color_image, intrinsics)
            pcds.append(pcd)
            if len(pcds) < 2:
                continue
            try:
                transforms.append(get_transform(pcds[-2], pcds[-1]))
            except Exception as ex:
                print(ex)
                pcds.pop()
                print("Chain broken")
            
    merge = pcds[0]
    for i, pcd in enumerate(pcds[1:]):
        for m in transforms[:i+1]:
            pcd.transform(m)
        merge = merge + pcd.select_by_index(pcd.remove_statistical_outlier(nb_neighbors=250, std_ratio=0.1)[1])
    merge = merge.select_by_index(merge.remove_statistical_outlier(nb_neighbors=250, std_ratio=1.0)[1])

    o3d.visualization.draw_geometries([merge])
    pcd = merge
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
