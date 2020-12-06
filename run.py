import argparse
import logging
import numpy as np
import open3d as o3d
from scanner3d.registration.group.pose_graph import PoseGraphReg
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
    scanner.filter()
    mesh = scanner.generate_mesh()
    o3d.visualization.draw_geometries([mesh])


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
