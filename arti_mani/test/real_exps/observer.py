import time
from collections import defaultdict

import cv2
import numpy as np
import pyrealsense2 as rs
from arti_mani.utils.cv_utils import visualize_depth

color_cache = defaultdict(lambda: {})
WIDTH, HEIGHT = 640, 360
RS_WIDTH, RS_HEIGHT = 256, 144
RS_RATIO = 0.4
save_path = "./"


def get_depth_real():
    # ...from Camera 1
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device("918512071887")  # hand_camera d435
    config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # ...from Camera 2
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device("104122061117")  # base_camera d415
    config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming from both cameras
    pipeline_1.start(config_1)
    pipeline_2.start(config_2)

    # Camera 1
    # Wait for a coherent pair of frames: depth and color
    frames_1 = pipeline_1.wait_for_frames()
    depth_frame_1 = frames_1.get_depth_frame()

    # Convert images to numpy arrays
    depth_image_1 = np.asanyarray(depth_frame_1.get_data())
    depth_image_1_small = cv2.resize(depth_image_1, (320, 180))

    # Camera 2
    # Wait for a coherent pair of frames: depth and color
    frames_2 = pipeline_2.wait_for_frames()
    depth_frame_2 = frames_2.get_depth_frame()
    depth_image_2 = np.asanyarray(depth_frame_2.get_data())
    depth_image_2_small = cv2.resize(depth_image_2, (320, 180))
    print(depth_image_1_small.shape, depth_image_2_small.shape)
    depth_images = np.stack((depth_image_1_small, depth_image_2_small), axis=2)
    return depth_images


def depth2xyz(depth_map, depth_cam_matrix, flatten=True, depth_scale=0.0010):
    """
    depth_map = np.random.randint(0, 10000,(720, 1280))
    depth_cam_matrix = np.array([[540, 0,  640],
                                 [0,   540,360],
                                 [0,   0,    1]])
    pc = depth2xyz(depth_map, depth_cam_matrix)
    """
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0 : depth_map.shape[0], 0 : depth_map.shape[1]]
    z = depth_map * depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy

    xyz = np.dstack((x, y, z)).reshape(-1, 3) if flatten else np.dstack((x, y, z))
    return xyz  # [N,3]


class Observer:
    def __init__(self, base_camera_SN="104122061117", mode="depth") -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(base_camera_SN)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, 30)

        # set align mode
        align_to = rs.stream.color  # depth align to color
        # align_to = rs.stream.depth  # color align to depth
        self.alignedFs = rs.align(align_to)

        self.cfg = self.pipeline.start(config)
        time.sleep(2)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.cfg.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.exposure, 5000.000)
        print("depth sensor exposure: ", depth_sensor.get_option(rs.option.exposure))

        color_sensor = self.cfg.get_device().first_color_sensor()
        color_sensor.set_option(rs.option.exposure, 300.000)
        print("color sensor exposure: ", color_sensor.get_option(rs.option.exposure))

        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)

        self.mode = mode

        ###################################################################################
        # realsense-viewer, orders of the postprocess:
        # decimation_filter --> HDR Merge --> threshold_filter --> Depth to Disparity --> spatial_filter
        # --> temporal_filter --> Disparity to Depth
        g_rs_downsample_filter = rs.decimation_filter(magnitude=1)  # 2 ** 1,  # 下采样率
        g_rs_thres_filter = rs.threshold_filter(min_dist=0.2, max_dist=2.0)
        g_rs_depth2disparity_trans = rs.disparity_transform(True)
        g_rs_spatical_filter = rs.spatial_filter(
            magnitude=2,
            smooth_alpha=0.5,
            smooth_delta=20,
            hole_fill=0,
        )
        g_rs_templ_filter = rs.temporal_filter(
            smooth_alpha=0.1, smooth_delta=40.0, persistence_control=3
        )
        g_rs_disparity2depth_trans = rs.disparity_transform(False)
        self.g_rs_depth_postprocess_list = [
            g_rs_downsample_filter,
            g_rs_thres_filter,
            g_rs_depth2disparity_trans,
            g_rs_spatical_filter,
            g_rs_templ_filter,
            g_rs_disparity2depth_trans,
        ]
        ###################################################################################

        self.colorizer = rs.colorizer()

    def get_observation(self):
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.alignedFs.process(frames)
                frame_num = aligned_frames.frame_number
                timestamp = aligned_frames.timestamp
                print("timestramp: ", timestamp)
                print("frame_num: ", frame_num)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_frame_filter = depth_frame
                for trans in self.g_rs_depth_postprocess_list:
                    depth_frame_filter = trans.process(depth_frame_filter)
                depth_frame = depth_frame_filter

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                colorized_depth = np.asanyarray(
                    self.colorizer.colorize(depth_frame).get_data()
                )

                print(color_image.shape, depth_image.shape, colorized_depth.shape)
                print(color_image.dtype, depth_image.dtype, colorized_depth.dtype)
                print(color_image.max(), depth_image.max(), colorized_depth.max())
                print(color_image.min(), depth_image.min(), colorized_depth.min())

                # resize color and depth
                rs_color_image = cv2.resize(
                    color_image, (RS_WIDTH, RS_HEIGHT), interpolation=cv2.INTER_NEAREST
                )
                rs_depth_image = cv2.resize(
                    depth_image, (RS_WIDTH, RS_HEIGHT), interpolation=cv2.INTER_NEAREST
                )

                rs_depth_colormap = visualize_depth(rs_depth_image)

                print(
                    rs_color_image.shape,
                    rs_depth_image.shape,
                    colorized_depth.shape,
                    rs_depth_colormap.shape,
                )
                print(
                    rs_color_image.dtype,
                    rs_depth_image.dtype,
                    colorized_depth.dtype,
                    rs_depth_colormap.dtype,
                )
                print(
                    rs_color_image.max(),
                    rs_depth_image.max(),
                    colorized_depth.max(),
                    rs_depth_colormap.max(),
                )
                print(
                    rs_color_image.min(),
                    rs_depth_image.min(),
                    colorized_depth.min(),
                    rs_depth_colormap.min(),
                )

                if self.mode == "rgbd":
                    frame = np.concatenate(
                        [rs_color_image, rs_depth_image[..., None] * self.depth_scale],
                        axis=2,
                    ).astype(np.float32)
                elif self.mode == "rgb":
                    frame = rs_color_image
                elif self.mode == "depth":
                    frame = rs_depth_image[..., None] * self.depth_scale
                else:
                    raise NotImplemented

                depth_intrin = (
                    depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
                )

                extrinsics = depth_frame.get_profile().get_extrinsics_to(
                    color_frame.get_profile()
                )
                rotation, translation = extrinsics.rotation, extrinsics.translation
                print("extrinsics: ")
                print(rotation, translation)

                rs_g_depth_intrinsics_matrix = np.array(
                    [
                        [depth_intrin.fx * RS_RATIO, 0.0, depth_intrin.ppx * RS_RATIO],
                        [0.0, depth_intrin.fy * RS_RATIO, depth_intrin.ppy * RS_RATIO],
                        [0, 0, 1.0],
                    ]
                )
                # print(rs_g_depth_intrinsics_matrix)
                points_xyz = depth2xyz(
                    rs_depth_image, rs_g_depth_intrinsics_matrix, self.depth_scale
                )
                # map to color frame (use extrinsics)
                points_xyz = points_xyz @ np.array(rotation).reshape(
                    (3, 3)
                ).transpose() + np.array(translation)

                print(points_xyz.shape, points_xyz.min(), points_xyz.max())

                return (
                    frame,
                    colorized_depth,
                    rs_depth_colormap,
                    points_xyz,
                    rs_g_depth_intrinsics_matrix,
                )

        finally:
            # self.pipeline.stop()
            pass
