import os
import copy
import argparse
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import logging
from tqdm import tqdm
from functools import partial

from scanner3d.camera import Camera
from scanner3d.registration.filterreg_pair_reg import FilterReg
from scanner3d.registration.pose_graph_reg import PoseGraphReg
from scanner3d.scanners.live_scanner import LiveScanner
from scanner3d.scanners.step_scanner import StepScanner

import faulthandler
faulthandler.enable()


def main(args):
    logging.basicConfig(level=args.log_level)
    if args.mode == "live":
        scanner = LiveScanner(args.log_level, FilterReg())
        scanner.start()
    else:
        scanner = StepScanner(args.log_level, PoseGraphReg(0.01))
        scanner.start()
    
    """
    pcds = [o3d.io.read_point_cloud(os.path.join("clouds", f)) for f in os.listdir("clouds")]

    o3d.visualization.draw_geometries(pcds)

    print(pcds)

    # Il ne s'agit pas d'une approche optimale puisqu'elle est O(N^2)
    # Toutefois elle est simple à comprendre.
    for i in range(len(pcds)):
        pcd = pcds[i]
        pcd = pcd.select_by_index(pcd.remove_statistical_outlier(nb_neighbors=500, std_ratio=0.1)[1])
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcds[i] = pcd

    o3d.visualization.draw_geometries(pcds)

    reg_type = "fpfh"

    print(len(pcds))
    meta_iter = 0
    while len(pcds) > 1:
        scores = []
        for i, pcd1 in enumerate(pcds):
            for j, pcd2 in enumerate(pcds):
                if i >= j:
                    continue
                if reg_type == "svr":
                    f_reg = l2dist_regs.registration_svr(pcd1, pcd2)
                    scores.append((i, j, 0, f_reg))
                elif reg_type == "filterreg":
                    f_reg = filterreg.registration_filterreg(pcd1, pcd2, tol=0.00001, maxiter=500, objective_type="pt2pl", feature_fn=features.FPFH())
                    scores.append((i, j, f_reg.q, f_reg.transformation))
                elif reg_type == "cpd":
                    f_reg = cpd.registration_cpd(pcd1, pcd2, tf_type_name="affine")
                    scores.append((i, j, f_reg.q, f_reg.transformation))
                elif reg_type == "icp":
                    f_reg = o3d.registration.registration_icp(pcd1, pcd2, 0.1, np.identity(4), o3d.registration.TransformationEstimationPointToPlane())
                    scores.append((i, j, f_reg.fitness, f_reg.transformation))
                # Fast Point Feature Histogram
                elif reg_type == "fpfh":
                    f_reg = o3d.registration.registration_fast_based_on_feature_matching(
                        pcd1,
                        pcd2,
                        o3d.registration.compute_fpfh_feature(pcd1, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)),
                        o3d.registration.compute_fpfh_feature(pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
                    )
                    scores.append((i, j, f_reg.fitness, f_reg.transformation))
                else:
                    raise Exception("Unknown registration algorithm")
                print(scores[-1][0:3])
        num_list = [i for i in range(len(pcds))]
        scores.sort(key=lambda x: x[2])
        new_pcds = []
        while num_list and scores:
            i, j, fit, transformation = scores.pop()

            if i not in num_list and j not in num_list:
                continue
            if fit < 0.80:
                continue

            target = copy.deepcopy(pcds[j])
            source = copy.deepcopy(pcds[i])
            if isinstance(transformation, np.ndarray):
                source.transform(transformation)
            else:
                source.points = transformation.transform(source.points)
            source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            new_pcd = target + source
            new_pcd = new_pcd.voxel_down_sample(voxel_size=0.01)
            new_pcds.append(new_pcd)
            print(i, j, fit)
            target2 = copy.deepcopy(target)
            source2 = copy.deepcopy(source)
            source2.paint_uniform_color([1, 0.706, 0])
            target2.paint_uniform_color([0, 0.651, 0.929])
            print(len(new_pcds) - 1)
            o3d.visualization.draw_geometries([source2, target2])
            o3d.io.write_point_cloud(f"new_clouds/{meta_iter}_{i}_{j}_{fit}.pcd", pcd)
            try: num_list.remove(i)
            except: pass
            try: num_list.remove(j)
            except: pass
        pcds = new_pcds
        meta_iter += 1


    o3d.visualization.draw_geometries(pcds)
    pcd = pcds[0]
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
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scan a 3D object and save the mesh file")
    parser.add_argument(
        "--mode", type=str, nargs="?", help="Mode, should be one of live or step", default="step"
    )
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
    parser.add_argument(
        "--log-level", type=str, nargs="?", help="Python loglevel to use should be one of (DEBUG, INFO, WARNING, ERROR)", default="WARNING"
    )
    main(parser.parse_args())
