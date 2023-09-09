from typing import Dict

import numpy as np
import sapien.core as sapien
import sklearn
import transforms3d
import trimesh
from arti_mani.utils.o3d_utils import is_o3d, np2mesh, to_o3d
from arti_mani.utils.trimesh_utils import is_trimesh, to_trimesh
from sapien.core import Articulation, Pose
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import nearly_equivalent


def sample_on_unit_sphere(rng):
    """
    Algo from http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    v = np.zeros(3)
    while np.linalg.norm(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()
        v[2] = rng.normal()

    v = v / np.linalg.norm(v)
    return v


def sample_on_unit_circle(rng):
    v = np.zeros(2)
    while np.linalg.norm(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()

    v = v / np.linalg.norm(v)
    return v


def sample_grasp_points_ee(gripper_pos, x_offset=0.03):
    """
    sample 6 points representing gripper, in EE frame
    """
    x_trans = gripper_pos + 0.002
    finger_points = np.array(
        [
            [-0.14, 0, 0],
            [-0.07, 0, 0],
            [-0.07, x_trans, 0],
            [0, x_trans, 0],
            [-0.07, -x_trans, 0],
            [0, -x_trans, 0],
        ]
    )
    adjust_finger_points = finger_points + np.array([x_offset, 0, 0])
    return adjust_finger_points


def sample_grasp_multipoints_ee(gripper_pos, num_points_perlink=10, x_offset=0.03):
    """
    sample 6 points representing gripper, in EE frame
    """
    x_trans = gripper_pos + 0.002
    left_finger = np.linspace(
        [0, -x_trans, 0], [-0.07, -x_trans, 0], num_points_perlink
    )[:-1]
    left_knuckle = np.linspace([-0.07, -x_trans, 0], [-0.07, 0, 0], num_points_perlink)[
        :-1
    ]
    right_finger = np.linspace(
        [0, x_trans, 0], [-0.07, x_trans, 0], num_points_perlink
    )[:-1]
    right_knuckle = np.linspace([-0.07, x_trans, 0], [-0.07, 0, 0], num_points_perlink)[
        :-1
    ]
    base = np.linspace([-0.14, 0, 0], [-0.07, 0, 0], num_points_perlink)
    finger_points = np.concatenate(
        (base, left_knuckle, left_finger, right_knuckle, right_finger)
    )
    adjust_finger_points = finger_points + np.array([x_offset, 0, 0])
    return adjust_finger_points


def rotation_between_vec(a, b):  # from a to b
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)  # norm might be 0
    angle = np.arccos(a @ b)
    R = Rotation.from_rotvec(axis * angle)
    return R


def angle_between_vec(a, b):  # from a to b
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    angle = np.arccos(a @ b)
    return angle


def wxyz_to_xyzw(q):
    return np.concatenate([q[1:4], q[0:1]])


def xyzw_to_wxyz(q):
    return np.concatenate([q[3:4], q[0:3]])


def rotate_2d_vec_by_angle(vec, theta):
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_mat @ vec


def angle_distance(q0: sapien.Pose, q1: sapien.Pose):
    qd = (q0.inv() * q1).q
    return 2 * np.arctan2(np.linalg.norm(qd[1:]), qd[0])


def get_axis_aligned_bbox_for_articulation(art: Articulation):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    for link in art.get_links():
        lp = link.pose
        for s in link.get_collision_shapes():
            p = lp * s.get_local_pose()
            T = p.to_transformation_matrix()
            vertices = s.geometry.vertices * s.geometry.scale
            vertices = vertices @ T[:3, :3].T + T[:3, 3]
            mins = np.minimum(mins, vertices.min(0))
            maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs


def get_axis_aligned_bbox_for_actor(actor):
    mins = np.ones(3) * np.inf
    maxs = -mins

    for shape in actor.get_collision_shapes():  # this is CollisionShape
        scaled_vertices = shape.geometry.vertices * shape.geometry.scale
        local_pose = shape.get_local_pose()
        mat = (actor.get_pose() * local_pose).to_transformation_matrix()
        world_vertices = scaled_vertices @ (mat[:3, :3].T) + mat[:3, 3]
        mins = np.minimum(mins, world_vertices.min(0))
        maxs = np.maximum(maxs, world_vertices.max(0))

    return mins, maxs


def get_local_axis_aligned_bbox_for_link(link):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    for s in link.get_collision_shapes():
        p = s.get_local_pose()
        T = p.to_transformation_matrix()
        vertices = s.geometry.vertices * s.geometry.scale
        vertices = vertices @ T[:3, :3].T + T[:3, 3]
        mins = np.minimum(mins, vertices.min(0))
        maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    assert H.shape == (4, 4), H.shape
    assert pts.ndim == 2 and pts.shape[1] == 3, pts.shape
    return pts @ H[:3, :3].T + H[:3, 3]


def get_oriented_bounding_box_for_2d_points(
    points_2d: np.ndarray, resolution=0.0
) -> Dict:
    assert len(points_2d.shape) == 2 and points_2d.shape[1] == 2
    if resolution > 0.0:
        points_2d = np.round(points_2d / resolution) * resolution
        points_2d = np.unique(points_2d, axis=0)
    ca = np.cov(points_2d, y=None, rowvar=0, bias=1)

    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)

    # use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    ar = np.dot(points_2d, np.linalg.inv(tvect))

    # get the minimum and maximum x and y
    mina = np.min(ar, axis=0)
    maxa = np.max(ar, axis=0)
    half_size = (maxa - mina) * 0.5

    # the center is just half way between the min and max xy
    center = mina + half_size
    # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array(
        [
            center + [-half_size[0], -half_size[1]],
            center + [half_size[0], -half_size[1]],
            center + [half_size[0], half_size[1]],
            center + [-half_size[0], half_size[1]],
        ]
    )

    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the centerback
    corners = np.dot(corners, tvect)
    center = np.dot(center, tvect)

    return {"center": center, "half_size": half_size, "axes": vect, "corners": corners}


def pose2mat(pose):
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = transforms3d.quaternions.quat2mat(pose[3:])
    return mat


def mat2pose(mat: np.ndarray):
    quat = transforms3d.quaternions.mat2quat(mat[:3, :3])
    pos = mat[:3, 3]
    pose = np.concatenate((pos, quat))
    return pose


def pose2trans_axangle(pose):
    rot_mat = pose2mat(pose)
    axis, angle = transforms3d.axangles.mat2axangle(
        rot_mat[:3, :3]
    )  # (normalized pivot axis:3, angle:1)
    trans_axangle = np.concatenate((rot_mat[:3, 3], angle * axis))
    return trans_axangle


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


def euler2quat(rot_xyz, axes="sxyz", unit="deg"):  ## rx, ry, rz, axes='sxyz'
    if unit == "deg":
        rx, ry, rz = (
            rot_xyz[0] * np.pi / 180,
            rot_xyz[1] * np.pi / 180,
            rot_xyz[2] * np.pi / 180,
        )
    return transforms3d.euler.euler2quat(rx, ry, rz, axes)


def quat2euler(quat, axes="sxyz", unit="deg"):  ## rx, ry, rz, axes='sxyz'
    rot_xyz = np.array(transforms3d.euler.quat2euler(quat, axes))
    if unit == "deg":
        rot_xyz = rot_xyz * 180 / np.pi
    return rot_xyz


def all_close(p1, p2, t_eps=1e-4, r_eps=1e-5):
    assert type(p1) == type(p2)
    return np.allclose(p1.p, p2.p, atol=t_eps) and nearly_equivalent(
        p1.q, p2.q, atol=r_eps
    )


def convex_hull(x, o3d=True):
    x = to_trimesh(x)
    x = trimesh.convex.convex_hull(x)
    if o3d:
        x = to_o3d(x)
    return x


def apply_pose(pose, x):
    if x is None:
        return x
    if isinstance(pose, np.ndarray):
        pose = Pose.from_transformation_matrix(pose)
    assert isinstance(pose, Pose)
    if isinstance(x, Pose):
        return pose * x
    elif isinstance(x, np.ndarray):
        assert x.ndim == 2 and x.shape[1] == 3
        x = np.concatenate((x, np.ones((x.shape[0], 1), dtype=x.dtype)), axis=1)
        return (x @ pose.to_transformation_matrix().T)[:, :3]
    elif is_trimesh(x) or is_o3d(x):
        sign = is_o3d(x)
        x = to_trimesh(x)
        if isinstance(x, trimesh.Trimesh):
            vertices = x.vertices
            faces = x.faces
            vertices = apply_pose(pose, vertices)
            x = trimesh.Trimesh(vertices=vertices, faces=faces)
        elif isinstance(x, trimesh.points.PointCloud):
            vertices = x.vertices
            vertices = apply_pose(pose, vertices)
            x = trimesh.points.PointCloud(vertices=vertices)
        if sign:
            x = to_o3d(x)
        return x
    else:
        print(x, type(x))
        raise NotImplementedError("")


def check_coplanar(vertices):
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(vertices)
    return pca.singular_values_[-1] < 1e-3
