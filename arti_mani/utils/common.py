import gzip
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Union

import gym
import numpy as np
from arti_mani import ASSET_DIR
from gym import spaces
from gym.utils.colorize import colorize


# -------------------------------------------------------------------------- #
# Basic
# -------------------------------------------------------------------------- #
def merge_dicts(ds: List[Dict], asarray=False):
    ret = {k: [d[k] for d in ds] for k in ds[0].keys()}
    if asarray:
        ret = {k: np.concatenate(v) for k, v in ret.items()}
    return ret


def warn(message, stacklevel=2):
    warnings.warn(colorize(message, "yellow"), stacklevel=stacklevel)


# -------------------------------------------------------------------------- #
# Numpy
# -------------------------------------------------------------------------- #
def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)


def compute_angle_between(x1, x2):
    """Compute angle (radian) between two vectors."""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()


class np_random:
    """Context manager for numpy random state"""

    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)
        return self.state

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self.state)


def random_choice(x: Sequence, rng: np.random.RandomState = np.random):
    assert len(x) > 0
    if len(x) == 1:
        return x[0]
    else:
        return x[rng.choice(len(x))]


# ---------------------------------------------------------------------------- #
# OpenAI gym
# ---------------------------------------------------------------------------- #
def register_gym_env(name, **kwargs):
    """A decorator to register gym environments.

    Args:
        name (str): a unique id to register in gym.
    """

    def _register(cls):
        entry_point = "{}:{}".format(cls.__module__, cls.__name__)
        gym.register(name, entry_point=entry_point, **kwargs)
        return cls

    return _register


def register_gym_env_for_xpw(name, **kwargs):
    def _register(cls):
        entry_point = "{}:{}".format(cls.__module__, cls.__name__)
        for obs_mode in ["state", "rgbd", "depth"]:
            tag = obs_mode
            gym.register(
                id=name.replace("@", tag),
                entry_point=entry_point,
                kwargs={
                    "obs_mode": obs_mode,
                    "reward_mode": "dense",
                    "articulation_config_path": ASSET_DIR
                    / "partnet_mobility_configs/1999-cabinet-door-real-1.yml",
                },
                **kwargs,
            )
        return cls

    return _register


def convert_observation_to_space(observation):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from gym.envs.mujoco_env
    """
    if isinstance(observation, (dict)):
        if not isinstance(observation, OrderedDict):
            warn(
                "observation is not an OrderedDict. Keys are {}".format(
                    observation.keys()
                )
            )
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        if observation.dtype == np.uint8:
            low = np.zeros(observation.shape, dtype=np.uint8)
            high = np.ones(observation.shape, dtype=np.uint8) * 255
        else:
            low = np.full(observation.shape, -float("inf"))
            high = np.full(observation.shape, float("inf"))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def normalize_action_space(action_space: spaces.Box):
    assert isinstance(action_space, spaces.Box), type(action_space)
    return spaces.Box(-1, 1, shape=action_space.shape, dtype=action_space.dtype)


def clip_and_scale_action(action, low, high, reverse=False):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    low, high = np.asarray(low), np.asarray(high)
    action = np.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action


def normalize_action(action, low, high):
    """Normalize and clip action to [-1, 1] according to a range [low, high]."""
    low, high = np.asarray(low), np.asarray(high)
    action = 2.0 * (action - low) / (high - low) - 1.0
    return np.clip(action, -1.0, 1.0)


def flatten_state_dict(state_dict: OrderedDict):
    """Flatten an ordered dict containing states recursively.

    Args:
        state_dict (OrderedDict): an ordered dict containing states.

    Raises:
        TypeError: If @state_dict is not an OrderedDict.
        TypeError: If a value of @state_dict is a dict instead of OrderedDict.
        AssertionError: If a value of @state_dict is an ndarray with ndim > 1.

    Returns:
        np.ndarray: flattened states.
    """
    if not isinstance(state_dict, OrderedDict):
        raise TypeError(
            "Must be an OrderedDict, but received {}".format(type(state_dict))
        )
    if len(state_dict) == 0:
        return np.empty(0)
    states = []
    for key, value in state_dict.items():
        if isinstance(value, OrderedDict):
            states.append(flatten_state_dict(value))
        elif isinstance(value, dict):
            raise TypeError(
                "Must be an OrderedDict, but received dict for {}".format(key)
            )
        elif isinstance(value, (int, float, tuple, list)):
            states.append(value)
        elif isinstance(value, np.ndarray):
            assert value.ndim <= 1, "Too many dimensions({}) for {}".format(
                value.ndim, key
            )
            states.append(value)
        elif isinstance(value, np.bool_):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            states.append(value.astype(int))
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
    return np.hstack(states)


def convert_np_bool_to_float(x) -> np.ndarray:
    return np.array(x).astype(float)


def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret
