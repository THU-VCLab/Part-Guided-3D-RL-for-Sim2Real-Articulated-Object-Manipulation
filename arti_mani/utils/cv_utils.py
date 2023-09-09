import cv2
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def show_cmap(cmap, norm=None, extend=None):
    """show the colormap."""
    if norm is None:
        norm = mcolors.Normalize(vmin=0, vmax=cmap.N)
    im = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    fig.colorbar(im, cax=ax, orientation="horizontal", extend=extend)
    plt.show()


def depth2pts_np(
    depth_map: np.ndarray,
    cam_intrinsic: np.ndarray,
    cam_extrinsic: np.ndarray = np.eye(4),
) -> np.ndarray:
    assert (len(depth_map.shape) == 2) or (
        len(depth_map.shape) == 3 and depth_map.shape[2] == 1
    )
    assert cam_intrinsic.shape == (3, 3)
    assert cam_extrinsic.shape == (4, 4)
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))  # (3, N)

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()  # (N, 3)
    return world_points


def get_pixel_grids_np(height: int, width: int):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(np.float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid


def visualize_depth(depth):
    MAX_DEPTH = 2.0
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth


def get_cmaps(num_colors):
    if num_colors == 6:
        cmaps = [
            (1, 0, 0),  # "red"
            (0, 0, 1),  # "blue"
            (1, 1, 0),  # "yellow"
            (0, 1, 0),  # "green"
            (0.627, 0.125, 0.941),  # "purple"
            (0.753, 0.753, 0.753),  # "grey"
        ]
    elif num_colors == 5:
        cmaps = [
            (1, 0, 0),  # "red"
            (0, 0, 1),  # "blue"
            (1, 1, 0),  # "yellow"
            (0, 1, 0),  # "green"
            (0.753, 0.753, 0.753),  # "grey"
        ]
    else:
        cmaps = cm.get_cmap("jet", num_colors)(range(num_colors))[:, :3]
    return cmaps


def visualize_seg(seg, num_classes=6):
    H, W = seg.shape
    cmaps = get_cmaps(num_classes)
    segvis_mask = np.zeros((H, W, 3))
    for ind in range(num_classes):
        segvis_mask[seg == ind] = cmaps[ind]
    return segvis_mask


class OpenCVViewer:
    def __init__(self, name="OpenCVViewer", is_rgb=True, exit_on_esc=True):
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        self.is_rgb = is_rgb
        self.exit_on_esc = exit_on_esc

    def imshow(self, image: np.ndarray, is_rgb=None, non_blocking=False, delay=0):
        if image.ndim == 2:
            image = np.tile(image[..., np.newaxis], (1, 1, 3))
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        assert image.ndim == 3, image.shape

        if self.is_rgb or is_rgb:
            image = image[..., ::-1]
        cv2.imshow(self.name, image)

        if non_blocking:
            return
        else:
            key = cv2.waitKey(delay)
            if key == 27:  # escape
                if self.exit_on_esc:
                    exit(0)
                else:
                    return None
            elif key == -1:  # timeout
                pass
            else:
                return chr(key)

    def close(self):
        cv2.destroyWindow(self.name)

    def __del__(self):
        self.close()
