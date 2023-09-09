from typing import List

import numpy as np
import open3d as o3d
import sapien.core as sapien
import sklearn
import trimesh


def check_coplanar(vertices):
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(vertices)
    return pca.singular_values_[-1] < 1e-3


# ---------------------------------------------------------------------------- #
# Convert in opne3d
# ---------------------------------------------------------------------------- #


def get_visual_body_meshes(visual_body: sapien.RenderBody):
    meshes = []
    for render_shape in visual_body.get_render_shapes():
        vertices = render_shape.mesh.vertices * visual_body.scale  # [n, 3]
        faces = render_shape.mesh.indices.reshape(-1, 3)  # [m * 3]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.apply_transform(visual_body.local_pose.to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_actor_visual_meshes(actor: sapien.ActorBase):
    """Get actor (visual) meshes in the actor frame."""
    meshes = []
    for vb in actor.get_visual_bodies():
        meshes.extend(get_visual_body_meshes(vb))
    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def merge_mesh(meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    if not meshes:
        return None
    # Merge without color and normal
    vertices = np.zeros((0, 3))
    triangles = np.zeros((0, 3))

    for mesh in meshes:
        vertices_i = np.asarray(mesh.vertices)
        triangles_i = np.asarray(mesh.triangles)
        triangles_i += vertices.shape[0]
        vertices = np.append(vertices, vertices_i, axis=0)
        triangles = np.append(triangles, triangles_i, axis=0)

    vertices = o3d.utility.Vector3dVector(vertices)
    triangles = o3d.utility.Vector3iVector(triangles)
    # print(vertices, triangles)
    # exit(0)
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    mesh.compute_vertex_normals(normalized=True)
    mesh.compute_triangle_normals(normalized=True)
    return mesh


def get_actor_meshes(actor: sapien.ActorBase):
    """Get actor (collision) meshes in the actor frame."""
    meshes = []
    for col_shape in actor.get_collision_shapes():
        geom = col_shape.geometry
        if isinstance(geom, sapien.BoxGeometry):
            mesh = trimesh.creation.box(extents=2 * geom.half_lengths)
        elif isinstance(geom, sapien.CapsuleGeometry):
            mesh = trimesh.creation.capsule(
                height=2 * geom.half_length, radius=geom.radius
            )
        elif isinstance(geom, sapien.SphereGeometry):
            mesh = trimesh.creation.icosphere(radius=geom.radius)
        elif isinstance(geom, sapien.PlaneGeometry):
            continue
        elif isinstance(
            geom, (sapien.ConvexMeshGeometry, sapien.NonconvexMeshGeometry)
        ):
            vertices = geom.vertices  # [n, 3]
            faces = geom.indices.reshape(-1, 3)  # [m * 3]
            vertices = vertices * geom.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise TypeError(type(geom))
        mesh.apply_transform(col_shape.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_actor_mesh(actor: sapien.ActorBase, to_world_frame=True):
    mesh = merge_meshes(get_actor_meshes(actor))
    if mesh is None:
        return None
    if to_world_frame:
        T = actor.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_actor_visual_mesh(actor: sapien.ActorBase):
    mesh = merge_meshes(get_actor_visual_meshes(actor))
    if mesh is None:
        return None
    return mesh


def get_articulation_meshes(
    articulation: sapien.ArticulationBase, exclude_link_names=()
):
    """Get link meshes in the world frame."""
    meshes = []
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        mesh = get_actor_mesh(link, True)
        if mesh is None:
            continue
        meshes.append(mesh)
    return meshes


def mesh2pcd(mesh, sample_density, num_points=None) -> o3d.geometry.PointCloud:
    pcd_tmp = mesh.sample_points_uniformly(number_of_points=sample_density)
    points = np.asarray(pcd_tmp.points)
    normals = np.asarray(pcd_tmp.normals)

    pcd = o3d.geometry.PointCloud()
    if num_points:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        idx = idx[:num_points]
        points = points[idx]
        normals = normals[idx]
    # print(vertices.shape, normals.shape)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# ---------------------------------------------------------------------------- #
# Build from numpy
# ---------------------------------------------------------------------------- #


def np2mesh(
    vertices, triangles, colors=None, vertex_normals=None, triangle_normals=None
) -> o3d.geometry.TriangleMesh:
    """Convert numpy array to open3d TriangleMesh."""
    vertices = o3d.utility.Vector3dVector(vertices)
    triangles = o3d.utility.Vector3iVector(triangles)
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(vertices)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(vertices), 1))
        else:
            raise RuntimeError(colors.shape)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if vertex_normals is not None:
        assert len(triangles) == len(vertex_normals)
        mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    else:
        mesh.compute_vertex_normals(normalized=True)

    if triangle_normals is not None:
        assert len(triangles) == len(triangle_normals)
        mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)
    else:
        mesh.compute_triangle_normals(normalized=True)
    return mesh


def np2pcd(points, colors=None, normals=None) -> o3d.geometry.PointCloud:
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def create_aabb(bbox, color=(0, 1, 0)) -> o3d.geometry.AxisAlignedBoundingBox:
    """Draw an axis-aligned bounding box."""
    assert (
        len(bbox) == 6
    ), f"The format of bbox should be xyzwlh, but received {len(bbox)}."
    bbox = np.asarray(bbox)
    abb = o3d.geometry.AxisAlignedBoundingBox(
        bbox[0:3] - bbox[3:6] * 0.5, bbox[0:3] + bbox[3:6] * 0.5
    )
    abb.color = color
    return abb


def create_aabb_from_pcd(
    pcd: np.ndarray, color=(0, 1, 0)
) -> o3d.geometry.AxisAlignedBoundingBox:
    """Draw an axis-aligned bounding box."""
    assert (
        pcd.shape[-1] == 3
    ), f"The format of bbox should be xyzwlh, but received {pcd.shape}."
    abb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(pcd)
    )
    abb.color = color
    return abb


def create_obb(bbox, R, color=(0, 1, 0)):
    """Draw an oriented bounding box."""
    assert (
        len(bbox) == 6
    ), f"The format of bbox should be xyzwlh, but received {len(bbox)}."
    obb = o3d.geometry.OrientedBoundingBox(bbox[0:3], R, bbox[3:6])
    obb.color = color
    return obb


def create_obb_from_pcd(pcd, color=(0, 1, 0)):
    """Draw an axis-aligned bounding box."""
    assert (
        pcd.shape[-1] == 3
    ), f"The format of bbox should be xyzwlh, but received {pcd.shape}."
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(pcd)
    )
    obb.color = color
    return obb


# ---------------------------------------------------------------------------- #
# Computation
# ---------------------------------------------------------------------------- #
def compute_pcd_normals(points, search_param=None, camera_location=(0.0, 0.0, 0.0)):
    """Compute normals."""
    pcd = np2pcd(points)
    if search_param is None:
        pcd.estimate_normals()
    else:
        pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_towards_camera_location(camera_location)
    normals = np.array(pcd.normals)
    return normals


def pcd_voxel_down_sample_with_crop(
    points,
    voxel_size,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> List[int]:
    """Crop and downsample the point cloud and return sample indices."""
    crop_mask = np.logical_and(
        np.logical_and.reduce(points > min_bound[None, :], axis=1),
        np.logical_and.reduce(points < max_bound[None, :], axis=1),
    )
    if not crop_mask.any():
        return []
    else:
        crop_indices = np.where(crop_mask)[0]
        points = points[crop_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        downsample_pcd, mapping, index_buckets = pcd.voxel_down_sample_and_trace(
            voxel_size, min_bound[:, None], max_bound[:, None]
        )
        sample_indices = [crop_indices[int(x[0])] for x in index_buckets]
        return sample_indices


def pcd_uni_down_sample_with_crop(
    points,
    num,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> List[int]:
    """Crop and downsample the point cloud and return sample indices."""
    crop_mask = np.logical_and(
        np.logical_and.reduce(points > min_bound[None, :], axis=1),
        np.logical_and.reduce(points < max_bound[None, :], axis=1),
    )
    if not crop_mask.any():
        return []
    else:
        crop_indices = np.where(crop_mask)[0]
        sample_points = points[crop_mask]

        N = sample_points.shape[0]
        index = np.arange(N)
        if N > num:
            np.random.shuffle(index)
            index = index[:num]
        else:
            num_repeat = num // N
            index = np.concatenate([index for i in range(num_repeat)])
            index = np.concatenate([index, index[: num - N * num_repeat]])
        sample_indices = crop_indices[index]
        return sample_indices.tolist()


def is_o3d(x):
    return isinstance(x, (o3d.geometry.TriangleMesh, o3d.geometry.PointCloud))


def to_o3d(x):
    """
    Numpy support is for pcd!
    """
    if is_o3d(x):
        return x
    elif isinstance(x, np.ndarray):
        assert x.ndim == 2 and x.shape[-1] == 3
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x))
    elif isinstance(x, trimesh.Trimesh):
        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(x.vertices), o3d.utility.Vector3iVector(x.faces)
        )
    elif isinstance(x, trimesh.points.PointCloud):
        return o3d.geometry.PointCloud(x.vertices)
    else:
        print(type(x))
        raise NotImplementedError()


def draw_3d_example(env, obs):
    pcd_points = obs["pointcloud"]["xyz"]
    pcd_points = env.unwrapped._get_obs_pointcloud_gt()["pointcloud"]["xyz"]
    pt0 = pcd_points[:300]
    pt1 = pcd_points[300:600]
    pt2 = pcd_points[600:900]
    pt3 = pcd_points[900:1200]
    pt4 = env.unwrapped.get_obs()["extra"]["point_target"][None]
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pt0)
    pcd0.paint_uniform_color([1, 0, 0])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pt1)
    pcd1.paint_uniform_color([0, 1, 0])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pt2)
    pcd2.paint_uniform_color([0, 0, 1])
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(pt3)
    pcd3.paint_uniform_color([0, 1, 1])
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(pt4)
    pcd4.paint_uniform_color([1, 0, 1])
    XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd0, pcd1, pcd2, pcd3, pcd4, XYZ])

    pcd_points = obs["pointcloud"]["xyz"]
    pt0 = pcd_points[:300]
    pt1 = pcd_points[300:600]
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pt0)
    pcd0.paint_uniform_color([1, 0, 0])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pt1)
    pcd1.paint_uniform_color([0, 1, 0])
    XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd0, pcd1, XYZ])

    import open3d as o3d
    from arti_mani.agents.camera import get_texture
    from arti_mani.utils.contrib import apply_pose_to_points
    from sapien.core import Pose

    pcgt = obs_global["pointcloud"]["xyz"][:300]
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pcgt)
    pcd0.paint_uniform_color([1, 0, 0])
    ego_cam = env.unwrapped._agent._cameras["hand_camera"]
    ego_cam.take_picture()
    position = get_texture(ego_cam, "Position")
    cam_xyz = position[..., :3]
    H, W, _ = cam_xyz.shape
    pc_cam = cam_xyz.reshape(-1, 3)
    ego_cam_extrin = env.unwrapped.cam_para["hand_camera_extrinsic_base_frame"]
    pc_ee = pc_cam @ ego_cam_extrin[:3, :3].transpose() + ego_cam_extrin[:3, 3]
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc_ee)
    pcd1.paint_uniform_color([0, 1, 0])
    XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd0, pcd1, XYZ])

    tipm = o3d.geometry.TriangleMesh.create_sphere(0.003)
    tipm.translate(point_pos)
    tipm.paint_uniform_color([0, 0, 1])
    XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([tipm, XYZ])
