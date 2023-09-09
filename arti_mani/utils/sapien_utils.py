from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.quaternions import mat2quat


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    else:
        return x / norm


def vectorize_pose(pose: sapien.Pose):
    return np.hstack([pose.p, pose.q])


def get_actor_by_name(actors, names: Union[List[str], str]):
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


def make_actor_visible(actor: sapien.Actor, visible=True):
    for v in actor.get_visual_bodies():
        v.set_visibility(1.0 if visible else 0.0)


@contextmanager
def set_default_physical_material(
    material: sapien.PhysicalMaterial, scene: sapien.Scene
):
    """Set default physical material within the context.

    Args:
        material (sapien.PhysicalMaterial): physical material to use as default.
        scene (sapien.Scene): scene instance.

    Yields:
        sapien.PhysicalMaterial: original default physical material.

    Example:
        with set_default_physical_material(material, scene):
            ...
    """
    old_material = scene.default_physical_material
    scene.default_physical_material = material
    try:
        yield old_material
    finally:
        scene.default_physical_material = old_material


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


# -------------------------------------------------------------------------- #
# Contact
# -------------------------------------------------------------------------- #
def get_pairwise_contacts(
    contacts: List[sapien.Contact], actor0: sapien.ActorBase, actor1: sapien.ActorBase
) -> List[Tuple[sapien.Contact, bool]]:
    pairwise_contacts = []
    for contact in contacts:
        if contact.actor0 == actor0 and contact.actor1 == actor1:
            pairwise_contacts.append((contact, True))
        elif contact.actor0 == actor1 and contact.actor1 == actor0:
            pairwise_contacts.append((contact, False))
    return pairwise_contacts


def compute_total_impulse(contact_infos: List[Tuple[sapien.Contact, bool]]):
    total_impulse = np.zeros(3)
    for contact, flag in contact_infos:
        contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        # Impulse is applied on the first actor
        total_impulse += contact_impulse * (1 if flag else -1)
    return total_impulse


def get_pairwise_contact_impulse(
    contacts: List[sapien.Contact], actor0: sapien.ActorBase, actor1: sapien.ActorBase
):
    pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)
    total_impulse = compute_total_impulse(pairwise_contacts)
    return total_impulse


def get_actor_contacts(
    contacts: List[sapien.Contact], actor: sapien.ActorBase
) -> List[Tuple[sapien.Contact, bool]]:
    actor_contacts = []
    for contact in contacts:
        if contact.actor0 == actor:
            actor_contacts.append((contact, True))
        elif contact.actor1 == actor:
            actor_contacts.append((contact, False))
    return actor_contacts


def get_articulation_contacts(
    contacts: List[sapien.Contact],
    articulation: sapien.Articulation,
    excluded_actors: Optional[List[sapien.Actor]] = None,
    included_links: Optional[List[sapien.Link]] = None,
) -> List[Tuple[sapien.Contact, bool]]:
    articulation_contacts = []
    links = articulation.get_links()
    if excluded_actors is None:
        excluded_actors = []
    if included_links is None:
        included_links = links
    for contact in contacts:
        if contact.actor0 in included_links:
            if contact.actor1 in links:
                continue
            if contact.actor1 in excluded_actors:
                continue
            articulation_contacts.append((contact, True))
            # print(contact.actor0, contact.actor1)
        elif contact.actor1 in included_links:
            if contact.actor0 in links:
                continue
            if contact.actor0 in excluded_actors:
                continue
            articulation_contacts.append((contact, False))
            # print(contact.actor0, contact.actor1)
    return articulation_contacts


def compute_max_impulse_norm(contact_infos: List[Tuple[sapien.Contact, bool]]):
    max_impulse_norms = [0]
    for contact, flag in contact_infos:
        max_impulse_norm = max(
            [np.linalg.norm(point.impulse) for point in contact.points]
        )
        max_impulse_norms.append(max_impulse_norm)
    return max(max_impulse_norms)


def get_articulation_max_impulse_norm(
    contacts: List[sapien.Contact],
    articulation: sapien.Articulation,
    excluded_actors: Optional[List[sapien.Actor]] = None,
):
    articulation_contacts = get_articulation_contacts(
        contacts, articulation, excluded_actors
    )
    max_impulse_norm = compute_max_impulse_norm(articulation_contacts)
    return max_impulse_norm


# -------------------------------------------------------------------------- #
# Camera
# -------------------------------------------------------------------------- #
def sapien_pose_to_opencv_extrinsic(sapien_pose_matrix: np.ndarray) -> np.ndarray:
    sapien2opencv = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ex = sapien2opencv @ np.linalg.inv(sapien_pose_matrix)  # world -> camera

    return ex


def look_at(eye, target, up=(0, 0, 1)) -> sapien.Pose:
    """Get the camera pose in SAPIEN by the Look-At method.

    Note:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        The SAPIEN camera follows the convention: (forward, right, up) = (x, -y, z)
        while the OpenGL camera follows (forward, right, up) = (-z, x, y)
        Note that the camera coordinate system (OpenGL) is left-hand.

    Args:
        eye: camera location
        target: looking-at location
        up: a general direction of "up" from the camera.

    Raises:
        RuntimeError: @up and @right are both None.

    Returns:
        sapien.Pose: camera pose
    """
    forward = normalize_vector(np.array(target) - np.array(eye))
    up = normalize_vector(up)
    left = np.cross(up, forward)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    return sapien.Pose(p=eye, q=mat2quat(rotation))


def hex2rgba(h, correction=True):
    # https://stackoverflow.com/a/29643643
    h = h.lstrip("#")
    r, g, b = tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    rgba = np.array([r, g, b, 1])
    if correction:  # reverse gamma correction in sapien
        rgba = rgba**2.2
    return rgba


def set_render_material(material: sapien.RenderMaterial, **kwargs):
    for k, v in kwargs.items():
        if k == "color":
            material.set_base_color(v)
        else:
            setattr(material, k, v)
    return material


def set_articulation_render_material(articulation: sapien.Articulation, **kwargs):
    for link in articulation.get_links():
        for b in link.get_visual_bodies():
            for s in b.get_render_shapes():
                mat = s.material
                set_render_material(mat, **kwargs)
                s.set_material(mat)


def set_cabinet_render_material(articulation: sapien.Articulation, **kwargs):
    handle_kwargs = {
        "color": kwargs.pop("color"),
        "diffuse_texture_filename": kwargs.pop("color_texture_filename"),
    }
    for link in articulation.get_links():
        for b in link.get_visual_bodies():
            if "handle" in b.get_name():
                for s in b.get_render_shapes():
                    mat = s.material
                    set_render_material(mat, **handle_kwargs)
                    s.set_material(mat)
            else:
                for s in b.get_render_shapes():
                    mat = s.material
                    set_render_material(mat, **kwargs)
                    s.set_material(mat)


def set_actor_render_material(actor: sapien.Actor, **kwargs):
    for b in actor.get_visual_bodies():
        for s in b.get_render_shapes():
            mat = s.material
            set_render_material(mat, **kwargs)
            s.set_material(mat)


def check_joint_stuck(
    articulation: sapien.Articulation,
    active_joint_idx: int,
    pos_diff_threshold: float = 1e-3,
    vel_threshold: float = 1e-4,
):
    actual_pos = articulation.get_qpos()[active_joint_idx]
    target_pos = articulation.get_drive_target()[active_joint_idx]
    actual_vel = articulation.get_qvel()[active_joint_idx]

    return (
        abs(actual_pos - target_pos) > pos_diff_threshold
        and abs(actual_vel) < vel_threshold
    )


def ignore_collision(articulation: sapien.Articulation):
    """ignore collision among all movable links for acceleration"""
    for joint, link in zip(articulation.get_joints(), articulation.get_links()):
        if joint.type in ["revolute", "prismatic"]:
            shapes = link.get_collision_shapes()
            for s in shapes:
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)
