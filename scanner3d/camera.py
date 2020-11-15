"""
Wraps the pyrealsense layer to return a format that Open3D can consume
"""
import logging
import pyrealsense2 as rs
import cv2
import numpy as np
import queue
import multiprocessing
import time
import open3d as o3d
from multiprocessing import Process, Manager
from scanner3d.exceptions import NoDeviceDetectedException


class Camera:
    def __init__(self, log_level):
        self.log_level = log_level
        self.manager = Manager()
        self.queue = self.manager.Queue()
        self.started = False
        self.last_color_image = None
        self.last_depth_image = None
        self.last_intrinsics = None
        self.camera_process = None

    def pcd(self):
        was_updated = False
        while True:
            try:
                (
                    self.last_color_image,
                    self.last_depth_image,
                    self.last_intrinsics,
                ) = self.queue.get(block=False)
                was_updated = True
            except queue.Empty:
                # We are up to date
                break

        if not was_updated:
            logging.warning("Capture was not updated for this pcd() call!")

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(self.last_color_image),
            o3d.geometry.Image(self.last_depth_image),
            convert_rgb_to_intensity=False,
        )
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.last_intrinsics
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, pinhole_camera_intrinsic
        )
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd

    def image_depth(self):
        if not self.started:
            self.start()
        while True:
            try:
                (
                    self.last_color_image,
                    self.last_depth_image,
                    self.last_intrinsics,
                ) = self.queue.get(block=(self.last_color_image is None))
                was_updated = True
            except queue.Empty:
                # We are up to date
                break
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.last_depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        return self.last_color_image, depth_colormap

    def start(self):
        # Reset all devices
        self.started = True
        self.camera_process = Process(
            target=Camera.loop, args=(self.queue, self.log_level)
        )
        self.camera_process.start()

    def stop(self):
        self.started = False
        self.camera_process.kill()

    def loop(q, log_level):
        logging.basicConfig(level=log_level)
        logging.debug("Camera process started")
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise NoDeviceDetectedException()
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

        while True:
            frames = aligner.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            logging.debug("Enqueuing new frame")
            q.put(
                (
                    color_image,
                    depth_image,
                    (
                        intrinsics.width,
                        intrinsics.height,
                        intrinsics.fx,
                        intrinsics.fy,
                        intrinsics.ppx,
                        intrinsics.ppy,
                    ),
                )
            )
            time.sleep(0.1)
