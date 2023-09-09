import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import sapien.core as sapien

try:
    import cupy
except ImportError as e:
    print("cupy is not installed.")


def get_texture(camera: sapien.CameraEntity, name, dtype="float"):
    """Get texture from camera.
    Faster data communication are supported by cupy.

    Args:
        camera (sapien.CameraEntity): SAPIEN camera
        name (str): texture name
        dtype (str, optional): texture dtype, [float/uint32].
            Defaults to "float".

    Returns:
        np.ndarray: texture
    """
    if os.environ.get("WITH_CUPY", 0) == 1:
        # dtype info is included already in the dl_tensor
        dlpack = camera.get_dl_tensor(name)
        return cupy.asnumpy(cupy.from_dlpack(dlpack))
    else:
        if dtype == "float":
            return camera.get_float_texture(name)
        elif dtype == "uint32":
            return camera.get_uint32_texture(name)
        else:
            raise NotImplementedError(f"Unsupported texture type: {dtype}")


def get_camera_rgb(camera: sapien.CameraEntity, uint8=True):
    image = get_texture(camera, "Color")[..., :3]  # [H, W, 3]
    if uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def get_camera_depth(camera: sapien.CameraEntity):
    # position texture is in OpenGL frame, and thus depth is negative.
    # The unit is meter
    depth = -get_texture(camera, "Position")[..., [2]]  # [H, W, 1]
    # mask = get_texture(camera, "Position")[..., [3]]
    # depth[mask == 1] = camera.far
    return depth


def get_camera_seg(camera: sapien.CameraEntity):
    seg = get_texture(camera, "Segmentation", "uint32")  # [H, W, 4]
    # channel 0 is visual id (mesh-level)
    # channel 1 is actor id (actor-level)
    return seg[..., :2]


def get_camera_images(
    camera: sapien.CameraEntity,
    rgb=True,
    depth=False,
    visual_seg=False,
    actor_seg=False,
) -> Dict[str, np.ndarray]:
    # Assume camera.take_picture() is called
    images = OrderedDict()
    if rgb:
        images["rgb"] = get_camera_rgb(camera)
    if depth:
        images["depth"] = get_camera_depth(camera)
    if visual_seg or actor_seg:
        seg = get_camera_seg(camera)
        if visual_seg:
            images["visual_seg"] = seg[..., 0]
        if actor_seg:
            images["actor_seg"] = seg[..., 1]
    return images


def get_camera_pcd(
    camera: sapien.CameraEntity,
    rgb=True,
    visual_seg=False,
    actor_seg=False,
) -> Dict[str, np.ndarray]:
    pcd = OrderedDict()
    # Each pixel is (x, y, z, z_buffer_depth) in OpenGL camera space
    position = get_texture(camera, "Position")  # [H, W, 4]
    # Remove invalid points
    mask = position[..., -1] < 1
    pcd["xyz"] = position[..., :3][mask]
    if rgb:
        pcd["rgb"] = get_camera_rgb(camera)[mask]
    if visual_seg or actor_seg:
        seg = get_camera_seg(camera)
        if visual_seg:
            pcd["visual_seg"] = seg[..., 0][mask]
        if actor_seg:
            pcd["actor_seg"] = seg[..., 1][mask]
    return pcd
