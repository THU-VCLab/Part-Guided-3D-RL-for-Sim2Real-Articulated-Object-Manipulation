import numpy as np
import open3d as o3d
import sapien.core as sapien
import trimesh
from arti_mani.utils.geometry import all_close, apply_pose, check_coplanar, convex_hull
from arti_mani.utils.o3d_utils import merge_mesh, np2mesh
from arti_mani.utils.string_utils import regex_match
from sapien.core import ActorBase, Articulation, CollisionShape, Pose
from transforms3d.euler import euler2mat


def get_actor_names(actors):
    single = False
    if isinstance(actors, ActorBase):
        actors = [
            actors,
        ]
        single = True
    elif isinstance(actors, Articulation):
        actors = actors.get_links()
    ret = [actor.get_name() for actor in actors]
    if single:
        ret = ret[0]
    return ret


def get_actors_by_names(actors, names):
    assert isinstance(actors, (list, tuple))
    # Actors can be joint and link
    if isinstance(names, str):
        names = [names]
        sign = True
    else:
        sign = False
    ret = [None for _ in names]
    for actor in actors:
        if actor.get_name() in names:
            ret[names.index(actor.get_name())] = actor
    return ret[0] if sign else ret


def collision_shape_to_o3d(
    collision_shape: CollisionShape,
) -> o3d.geometry.TriangleMesh:
    if collision_shape.type == "capsule":
        capsule = collision_shape.geometry
        mesh = trimesh.creation.capsule(
            capsule.half_length * 2, capsule.radius, count=[4, 4]
        )
        vertices = mesh.vertices
        indices = mesh.faces
        to_x = euler2mat(0, np.pi / 2, 0)
        vertices = apply_pose(to_x, vertices)
    elif collision_shape.type == "convex_mesh":
        convex = collision_shape.geometry
        vertices = convex.vertices * convex.scale
        vertices = apply_pose(Pose(p=[0, 0, 0], q=convex.rotation), vertices)
        indices = convex.indices.reshape(-1, 3)
    elif collision_shape.type == "sphere":
        sphere = collision_shape.geometry
        mesh = trimesh.creation.icosphere(radius=sphere.radius, subdivisions=1)
        vertices = mesh.vertices
        indices = mesh.faces
    elif collision_shape.type == "box":
        box = collision_shape.geometry
        mesh = trimesh.creation.box(extents=box.half_lengths * 2)
        vertices = mesh.vertices
        indices = mesh.faces
    else:
        print(collision_shape.type, "????")
        raise NotImplementedError("")
    vertices = apply_pose(collision_shape.get_local_pose(), vertices)
    vertices = o3d.utility.Vector3dVector(vertices)
    indices = o3d.utility.Vector3iVector(indices)
    return o3d.geometry.TriangleMesh(vertices=vertices, triangles=indices)


cached_o3d = {}


def actor_to_o3d(
    actor,
    convex=False,
    collision=False,
    pattern_name="(.*?)",
    actor_exclude=[],
    part_exclude=[],
    actor_name="",
    use_cache=True,
):
    if actor.get_name() in actor_exclude:
        return None
    ret = []
    is_actor = False
    actor_name_curr = actor_name + f"/{actor.get_name()}"
    if isinstance(actor, Articulation):
        for link in actor.get_links():
            tmp_o3d_mesh = actor_to_o3d(
                link,
                convex,
                collision,
                pattern_name,
                actor_exclude,
                part_exclude,
                actor_name=actor_name_curr,
                use_cache=use_cache,
            )
            if tmp_o3d_mesh:
                ret.append(tmp_o3d_mesh)
    elif isinstance(actor, ActorBase):
        is_actor = True
        if not collision:
            actor_name_curr = actor_name_curr + "_visual"
            print(list(cached_o3d.keys()))
            if actor_name_curr in cached_o3d and use_cache:
                ori_mesh, pose, trans_mesh = cached_o3d[actor_name_curr]
                print("Hit cache", use_cache, actor_name, actor.get_pose(), pose)
                if trans_mesh is None or all_close(pose, actor.get_pose()):
                    return trans_mesh
                ret = ori_mesh
            else:
                for visual_body in actor.get_visual_bodies():
                    if not regex_match(visual_body.get_name(), pattern_name):
                        continue
                    if visual_body.get_name() in part_exclude:
                        continue
                    if visual_body.type == "mesh":
                        scale = visual_body.scale
                    elif visual_body.type == "box":
                        scale = visual_body.half_lengths
                    elif visual_body.type == "sphere":
                        scale = visual_body.radius
                    else:
                        scale = 1.0
                    for i in visual_body.get_render_shapes():
                        if i.mesh.indices.reshape(-1, 3).shape[0] < 4 or check_coplanar(
                            i.mesh.vertices * scale
                        ):
                            continue
                        vertices = apply_pose(
                            visual_body.local_pose, i.mesh.vertices * scale
                        )
                        mesh = np2mesh(vertices, i.mesh.indices.reshape(-1, 3))
                        if convex:
                            mesh = convex_hull(mesh)
                        ret.append(mesh)
        else:
            actor_name_curr = actor_name_curr + "_collision"
            for collision_shape in actor.get_collision_shapes():
                mesh = collision_shape_to_o3d(collision_shape)
                ret.append(mesh)
    else:
        print(type(actor))
        raise NotImplementedError()
    # print(ret)
    if not ret:
        # raise ValueError(f"No entity in {actor}")
        return None
    if isinstance(ret, list):
        ret = merge_mesh(ret)
    if is_actor:
        num_faces = 1024
        if ret is not None and np.asarray(ret.triangles).shape[0] > num_faces:
            ret = ret.simplify_vertex_clustering(
                voxel_size=0.02,
                contraction=o3d.geometry.SimplificationContraction.Average,
            )
        trans_mesh = apply_pose(actor.get_pose(), ret)
        if actor_name_curr not in cached_o3d and use_cache:
            cached_o3d[actor_name_curr] = [ret, actor.get_pose(), trans_mesh]
        ret = trans_mesh
    return ret


def actor_to_o3d_mesh(
    actor: ActorBase,
    collision: bool = False,
    convex: bool = False,
    max_num_faces: int = 1024,
) -> o3d.geometry.TriangleMesh:
    assert isinstance(actor, ActorBase), type(actor)
    shape_list = []
    if collision:
        for collision_shape in actor.get_collision_shapes():
            mesh = collision_shape_to_o3d(collision_shape)
            shape_list.append(mesh)
    else:
        for visual_body in actor.get_visual_bodies():
            if visual_body.type == "mesh":
                scale = visual_body.scale
            elif visual_body.type == "box":
                scale = visual_body.half_lengths
            elif visual_body.type == "sphere":
                scale = visual_body.radius
            else:
                scale = 1.0
            for render_shape in visual_body.get_render_shapes():
                if render_shape.mesh.indices.reshape(-1, 3).shape[
                    0
                ] < 4 or check_coplanar(render_shape.mesh.vertices * scale):
                    continue
                vertices = apply_pose(
                    visual_body.local_pose, render_shape.mesh.vertices * scale
                )
                mesh = np2mesh(vertices, render_shape.mesh.indices.reshape(-1, 3))
                if convex:
                    mesh = convex_hull(mesh)
                shape_list.append(mesh)
    if not shape_list:
        return None
    mesh = merge_mesh(shape_list)
    if np.asarray(mesh.triangles).shape[0] > max_num_faces:
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=0.02, contraction=o3d.geometry.SimplificationContraction.Average
        )
    return mesh


def project_qpos_to_valid_range(robot, qpos):
    for i, joint in enumerate(robot.get_active_joints()):
        l, r = joint.get_limits()[0]
        if l <= qpos[i] <= r:
            continue
        if joint.type == "revolute":
            print(qpos[i], l, r, qpos[i] - l)

        elif joint.type == "prismatic":
            return None

    exit(0)


def get_entity_by_name(entities, name: str, is_unique=True):
    """Get a Sapien.Entity given the name.

    Args:
        entities (List[sapien.Entity]): entities (link, joint, ...) to query.
        name (str): name for query.
        is_unique (bool, optional):
            whether the name should be unique. Defaults to True.

    Raises:
        RuntimeError: The name is not unique when @is_unique is True.

    Returns:
        sapien.Entity or List[sapien.Entity]:
            matched entity or entities. None if no matches.
    """
    matched_entities = [x for x in entities if x.get_name() == name]
    if len(matched_entities) > 1:
        if not is_unique:
            return matched_entities
        else:
            raise RuntimeError(f"Multiple entities with the same name {name}.")
    elif len(matched_entities) == 1:
        return matched_entities[0]
    else:
        return None


# -------------------------------------------------------------------------- #
# Entity state
# -------------------------------------------------------------------------- #
def get_actor_state(actor: sapien.Actor):
    pose = actor.get_pose()
    if actor.type == "static":
        vel = np.zeros(3)
        ang_vel = np.zeros(3)
    else:
        vel = actor.get_velocity()  # [3]
        ang_vel = actor.get_angular_velocity()  # [3]
    return np.hstack([pose.p, pose.q, vel, ang_vel])  # 3+4+3+3=13


def set_actor_state(actor: sapien.Actor, state: np.ndarray):
    assert len(state) == 13, len(state)
    actor.set_pose(Pose(state[0:3], state[3:7]))
    if actor.type != "static":
        actor.set_velocity(state[7:10])
        actor.set_angular_velocity(state[10:13])


def get_articulation_state(articulation: sapien.Articulation):
    root_link = articulation.get_links()[0]
    pose = root_link.get_pose()
    vel = root_link.get_velocity()  # [3]
    ang_vel = root_link.get_angular_velocity()  # [3]
    qpos = articulation.get_qpos()
    qvel = articulation.get_qvel()
    return np.hstack(
        [pose.p, pose.q, vel, ang_vel, qpos, qvel]
    )  # 7+2 robot: 3+4+3+3+9+9=31


def set_articulation_state(articulation: sapien.Articulation, state: np.ndarray):
    articulation.set_root_pose(Pose(state[0:3], state[3:7]))
    articulation.set_root_velocity(state[7:10])
    articulation.set_root_angular_velocity(state[10:13])
    qpos, qvel = np.split(state[13:], 2)
    articulation.set_qpos(qpos)
    articulation.set_qvel(qvel)


def get_pad_articulation_state(
    articulation: sapien.Articulation, max_dof: int
) -> np.ndarray:
    root_link = articulation.get_links()[0]
    pose = root_link.get_pose()
    base_pos, base_quat = pose.p, pose.q
    base_vel = root_link.get_velocity()  # [3]
    base_ang_vel = root_link.get_angular_velocity()  # [3]
    qpos = articulation.get_qpos()
    qvel = articulation.get_qvel()
    k = len(qpos)
    # pad_obj_internal_state = np.zeros(max_dof)
    # pad_obj_internal_state[:k] = qpos
    pad_obj_internal_state = np.zeros(2 * max_dof)
    pad_obj_internal_state[:k] = qpos
    pad_obj_internal_state[max_dof : max_dof + k] = qvel
    return np.concatenate(
        [base_pos, base_quat, base_vel, base_ang_vel, pad_obj_internal_state]
    )


def get_actor_AABB(actor):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    if isinstance(actor, Articulation):
        for link in actor.get_links():
            mins_i, maxs_i = get_actor_AABB(link)
            mins = np.minimum(mins, mins_i)
            maxs = np.maximum(maxs, maxs_i)
    else:
        assert isinstance(actor, ActorBase)
        actor_pose = actor.get_pose()
        for s in actor.get_collision_shapes():
            p = actor_pose * s.get_local_pose()
            T = p.to_transformation_matrix()
            vertices = s.geometry.vertices * s.geometry.scale
            vertices = vertices @ T[:3, :3].T + T[:3, 3]
            mins = np.minimum(mins, vertices.min(0))
            maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs


def ignore_collision(articulation: sapien.Articulation):
    """ignore collision among all movable links for acceleration"""
    for joint, link in zip(articulation.get_joints(), articulation.get_links()):
        if joint.type in ["revolute", "prismatic"]:
            shapes = link.get_collision_shapes()
            for s in shapes:
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)
