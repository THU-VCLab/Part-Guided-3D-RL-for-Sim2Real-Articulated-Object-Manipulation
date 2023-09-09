import numpy as np
import sapien.core as sapien
import transforms3d


def normalize_image(image):
    v_max = np.max(image)
    v_min = np.min(image)
    return (image - v_min) / (v_max - v_min)


def mat2pose(mat: np.ndarray):
    quat = transforms3d.quaternions.mat2quat(mat[:3, :3])
    pos = mat[:3, 3]
    # pose = np.concatenate((pos, quat))
    pose = sapien.Pose(pos, quat)
    return pose


def trans_axangle2pose(trans_axangle):
    RT = np.eye(4)
    RT[:3, 3] = trans_axangle[:3]
    angle = np.linalg.norm(trans_axangle[3:6])
    if angle < 1e-6:
        axis = (0, 0, 1)
    else:
        axis = trans_axangle[3:6] / angle
    RT[:3, :3] = transforms3d.axangles.axangle2mat(axis, angle)
    return mat2pose(RT)
