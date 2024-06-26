"""Preprocessing for embedding motion capture/dannce data."""

import dm_control
import h5py
from dm_control.locomotion.walkers import rodent
from dm_control.locomotion.walkers import rescale
from dm_control.utils import transformations as tr
from dm_control import mjcf
import pickle
import mocap_preprocess
import numpy as np
import sys
import os
import argparse
from scipy.io import loadmat
from typing import Text, List, Tuple, Dict, Union, Optional, Sequence
import subprocess
import jax
from jax import numpy as jp
from flax import struct
from walker import Rat
from typing import Any


def process(
    stac_path: Text,
    save_file: Text,
    scale_factor: float = 0.9,
    start_step: int = 0,
    clip_length: int = 250,
    n_steps: int = None,
    max_qvel: float = 20.0,
    dt: float = 0.02,
    adjust_z_offset: float = 0.0,
    verbatim: bool = False,
    ref_steps: Tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    """Summary

    Args:
        stac_path (Text): Path to stac file containing reference.
        save_file (Text): Path to Folder in which to save hdf5 dataset.
        start_step (int, optional): First frame in rollout
        clip_length (int, optional): Length of clip
        max_qvel (float, optional): Max allowed qvelocity
        dt (float, optional): Timestep
        adjust_z_offset (float, optional): Z-offset in m
        verbatim (bool, optional): Process in verbatim mode.
        ref_steps (Tuple, optional): Reference steps.
    """
    # load qpos from file
    with open(stac_path, "rb") as file:
        d = pickle.load(file)
        mocap_qpos = np.array(d["qpos"])

    # load rodent mjcf
    root = mjcf.from_path("./assets/rodent.xml")
    rescale.rescale_subtree(
        root,
        scale_factor,
        scale_factor,
    )
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)

    """Extract featires from the reference qpos"""
    if n_steps is None:
        n_steps = mocap_qpos.shape[0]

    jax_paths = []
    max_reference_index = np.max(ref_steps) + 1
    with h5py.File(save_file, "w") as file:
        for start_step in range(start_step, start_step + n_steps, clip_length):
            print(f"start_step: {start_step}", flush=True)
            end_step = np.min(
                [start_step + clip_length + max_reference_index, start_step + n_steps]
            )
            mocap_features = get_mocap_features(
                mocap_qpos[start_step:end_step, :],
                walker,
                physics,
                max_qvel,
                dt,
                adjust_z_offset,
                verbatim,
            )

            mocap_features["scaling"] = np.array([])
            mocap_features["markers"] = np.array([])
            save_features(file, mocap_features, f"clip_{start_step}")
            jax_paths.append(
                save_dataclass_pickle(
                    f"{save_file[:-3]}_clip_{start_step}.p", mocap_features
                )
            )
    return jax_paths


def get_mocap_features(
    mocap_qpos: np.ndarray,
    walker: rodent.Rat,
    physics,
    max_qvel: float,
    dt: float,
    adjust_z_offset: float,
    verbatim: bool,
    null_xyr: bool = False,
    shift_position=None,
    shift_rotation=None,
):
    """Convert mocap_qpos to valid reference features.

    Args:
        mocap_qpos (np.ndarray): Array of qpos data
        walker (rodent.Rat): rodent walker
        physics (TYPE): Environment Physics instance.
        max_qvel (float): Maximum allowable q velocity.
        dt (float): Timestep between qpos frames.
        adjust_z_offset (float): Adjust Z position by this amount.
        verbatim (bool): If true, preprocess verbatim.
        null_xyr (bool, optional): Description
        shift_position (bool, optional): Amount by which to shift position.
        shift_rotation (bool, optional): Amount by which to shift the rotation.
    """
    # Clip the angles.
    joint_names = [b.name for b in walker.mocap_joints]
    joint_ranges = physics.bind(walker.mocap_joints).range
    min_angles = joint_ranges[:, 0]
    max_angles = joint_ranges[:, 1]
    angles = mocap_qpos[:, 7:]
    clipped_angles = np.clip(angles, min_angles, max_angles)
    indexes = np.where(angles != clipped_angles)
    if verbatim and indexes[0].size != 0:
        for i, j in zip(*indexes):
            if np.abs(angles[i, j] - clipped_angles[i, j]) >= 0.1:
                print(
                    "Step {} angle of {} clipped from {} to {}.".format(
                        i, joint_names[j], angles[i, j], clipped_angles[i, j]
                    )
                )
    mocap_qpos[:, 7:] = clipped_angles
    # Generate the mocap_features.
    mocap_features = {}
    mocap_features["position"] = []
    mocap_features["quaternion"] = []
    mocap_features["joints"] = []
    mocap_features["center_of_mass"] = []
    mocap_features["end_effectors"] = []
    mocap_features["velocity"] = []
    mocap_features["angular_velocity"] = []
    mocap_features["joints_velocity"] = []
    mocap_features["appendages"] = []
    mocap_features["body_positions"] = []
    mocap_features["body_quaternions"] = []
    feet_height = []
    walker_bodies = walker.mocap_tracking_bodies
    body_names = [b.name for b in walker_bodies]
    # print(len(walker_bodies), body_names)
    if adjust_z_offset:
        left_foot_index = body_names.index("foot_L")
        right_foot_index = body_names.index("foot_R")

    # Padding for velocity corner case.
    mocap_qpos = np.concatenate([mocap_qpos, mocap_qpos[-1, np.newaxis, :]], axis=0)
    # print(mocap_qpos.shape)
    qvel = np.zeros(len(mocap_qpos[0]) - 1)

    for n_frame, qpos in enumerate(mocap_qpos[:-1]):
        set_walker(
            physics,
            walker,
            qpos,
            qvel,
            null_xyr=null_xyr,
            position_shift=shift_position,
            rotation_shift=shift_rotation,
        )
        freejoint = walker.mjcf_model.find(
            "joint", "root"
        )  # mjcf.get_attachment_frame(walker.mjcf_model).freejoint
        root_pos = physics.bind(freejoint).qpos[:3].copy()
        mocap_features["position"].append(root_pos)
        root_quat = physics.bind(freejoint).qpos[3:].copy()
        mocap_features["quaternion"].append(root_quat)
        joints = np.array(physics.bind(walker.mocap_joints).qpos)
        mocap_features["joints"].append(joints)
        freejoint_frame = walker.mjcf_model.find(
            "body", "torso"
        )  # mjcf.get_attachment_frame(walker.mjcf_model)
        com = np.array(physics.bind(freejoint_frame).subtree_com)
        mocap_features["center_of_mass"].append(com)
        end_effectors = np.copy(
            walker.observables.end_effectors_pos(physics)[:]
        ).reshape(-1, 3)
        mocap_features["end_effectors"].append(end_effectors)
        if hasattr(walker.observables, "appendages_pos"):
            appendages = np.copy(walker.observables.appendages_pos(physics)[:]).reshape(
                -1, 3
            )
        else:
            appendages = np.copy(end_effectors)
        mocap_features["appendages"].append(appendages)
        xpos = physics.bind(walker_bodies).xpos.copy()
        mocap_features["body_positions"].append(xpos)
        xquat = physics.bind(walker_bodies).xquat.copy()
        mocap_features["body_quaternions"].append(xquat)
        if adjust_z_offset:
            feet_height += [xpos[left_foot_index][2], xpos[right_foot_index][2]]

    # Array
    mocap_features["position"] = np.array(mocap_features["position"])
    mocap_features["quaternion"] = np.array(mocap_features["quaternion"])
    mocap_features["joints"] = np.array(mocap_features["joints"])
    mocap_features["center_of_mass"] = np.array(mocap_features["center_of_mass"])
    mocap_features["end_effectors"] = np.array(mocap_features["end_effectors"])
    mocap_features["appendages"] = np.array(mocap_features["appendages"])
    mocap_features["body_positions"] = np.array(mocap_features["body_positions"])
    mocap_features["body_quaternions"] = np.array(mocap_features["body_quaternions"])

    print(mocap_features["position"].shape)
    # Offset vertically the qpos and xpos to ensure that the clip is aligned
    # with the floor. The heuristic uses the 10 lowest feet heights and
    # compensates for the thickness of the geoms.
    feet_height = np.sort(feet_height)
    if adjust_z_offset:
        z_offset = feet_height[:10].mean() - 0.006
    else:
        z_offset = 0
    mocap_qpos[:, 2] -= z_offset
    mocap_features["position"][:, 2] -= z_offset
    mocap_features["center_of_mass"][:, 2] -= z_offset
    mocap_features["body_positions"][:, :, 2] -= z_offset

    # Calculate qvel, clip.
    mocap_qvel = compute_velocity_from_kinematics(mocap_qpos, dt)
    vels = mocap_qvel[:, 6:]
    clipped_vels = np.clip(vels, -max_qvel, max_qvel)
    indexes = np.where(vels != clipped_vels)
    if verbatim and indexes[0].size != 0:
        for i, j in zip(*indexes):
            if np.abs(vels[i, j] - clipped_vels[i, j]) >= 0.1:
                print(
                    "Step {} velocity of {} clipped from {} to {}.".format(
                        i, joint_names[j], vels[i, j], clipped_vels[i, j]
                    )
                )
    mocap_qvel[:, 6:] = clipped_vels
    mocap_features["velocity"] = mocap_qvel[:, :3]
    mocap_features["angular_velocity"] = mocap_qvel[:, 3:6]
    mocap_features["joints_velocity"] = mocap_qvel[:, 6:]
    return mocap_features


def set_walker(
    physics,
    walker: rodent.Rat,
    qpos: np.ndarray,
    qvel: np.ndarray,
    offset: Union[float, List, np.ndarray] = 0.0,
    null_xyr: bool = False,
    position_shift=None,
    rotation_shift=None,
):
    """Set the freejoint and walker's joints angles and velocities.

    Args:
        physics (TYPE): Environment Physics instance.
        walker (rodent.Rat): Description
        qpos (np.ndarray): Description
        qvel (np.ndarray): Description
        offset (Union[float, List, np.ndarray], optional): xyz offset
        null_xyr (bool, optional): Description
        position_shift (TYPE, optional): Amount by which to shift position.
        rotation_shift (TYPE, optional): Amount by which to shift the rotation.
    """
    qpos = qpos.copy()
    if null_xyr:
        qpos[:3] = 0.0
        euler = tr.quat_to_euler(qpos[3:7], ordering="ZYX")
        euler[0] = 0.0
        quat = tr.euler_to_quat(euler, ordering="ZYX")
        qpos[3:7] = quat
    qpos[:3] += offset
    freejoint = walker.mjcf_model.find(
        "joint", "root"
    )  # mjcf.get_attachment_frame(walker.mjcf_model).freejoint
    physics.bind(freejoint).qpos = qpos[:7]
    physics.bind(freejoint).qvel = qvel[:6]
    physics.bind(walker.mocap_joints).qpos = qpos[7:]
    physics.bind(walker.mocap_joints).qvel = qvel[6:]
    if position_shift is not None or rotation_shift is not None:
        walker.shift_pose(
            physics,
            position=position_shift,
            quaternion=rotation_shift,
            rotate_velocity=True,
        )


def compute_velocity_from_kinematics(
    qpos_trajectory: np.ndarray, dt: float
) -> np.ndarray:
    """Computes velocity trajectory from position trajectory.

    Args:
        qpos_trajectory (np.ndarray): trajectory of qpos values T x ?
          Note assumes has freejoint as the first 7 dimensions
        dt (float): timestep between qpos entries

    Returns:
        np.ndarray: Trajectory of velocities.
    """
    qvel_translation = (qpos_trajectory[1:, :3] - qpos_trajectory[:-1, :3]) / dt
    qvel_gyro = []
    for t in range(qpos_trajectory.shape[0] - 1):
        normed_diff = tr.quat_diff(qpos_trajectory[t, 3:7], qpos_trajectory[t + 1, 3:7])
        normed_diff /= np.linalg.norm(normed_diff)
        qvel_gyro.append(tr.quat_to_axisangle(normed_diff) / dt)
    qvel_gyro = np.stack(qvel_gyro)
    qvel_joints = (qpos_trajectory[1:, 7:] - qpos_trajectory[:-1, 7:]) / dt
    return np.concatenate([qvel_translation, qvel_gyro, qvel_joints], axis=1)


# 13 features
@struct.dataclass
class ReferenceClip:
    angular_velocity: jp.ndarray
    appendages: jp.ndarray
    body_positions: jp.ndarray
    body_quaternions: jp.ndarray
    center_of_mass: jp.ndarray
    end_effectors: jp.ndarray
    joints: jp.ndarray
    joints_velocity: jp.ndarray
    markers: jp.ndarray
    position: jp.ndarray
    quaternion: jp.ndarray
    scaling: jp.ndarray
    velocity: jp.ndarray


class ClipCollection:
    """Dataclass representing a collection of mocap reference clips.
    Store all the clip ids for retrieving"""

    def __init__(
        self,
        ids: Sequence[Text], # this is the id of the specific clip
        start_steps: Optional[Sequence[int]] = None,
        end_steps: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[Union[int, float]]] = None,
    ):
        """Instantiate a ClipCollection."""
        self.ids = ids
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.weights = weights
        num_clips = len(self.ids)
        try:
            if self.start_steps is None:
                # by default start at the beginning
                self.start_steps = (0,) * num_clips
            else:
                assert len(self.start_steps) == num_clips

            # without access to the actual clip we cannot specify an end_steps default
            if self.end_steps is not None:
                assert len(self.end_steps) == num_clips

            if self.weights is None:
                self.weights = (1.0,) * num_clips
            else:
                assert len(self.weights) == num_clips
                assert jp.all(np.array(self.weights) >= 0.0)
        except AssertionError as e:
            raise ValueError("ClipCollection validation failed. {}".format(e))


def save_dataclass_pickle(pickle_path, mocap_features):
    data = ReferenceClip(**mocap_features)
    data = jax.tree_map(lambda x: jp.array(x), data)
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)
    return pickle_path


def save_features(file: h5py.File, mocap_features: Dict, clip_name: Text):
    """Save features to hdf5 dataset

    Args:
        file (h5py.File): Hdf5 dataset
        mocap_features (Dict): Features extracted through rollout
        clip_name (Text): Name of the clip stored in the hdf5 dataset.
    """
    clip_group = file.create_group(clip_name)
    n_steps = len(mocap_features["center_of_mass"])
    clip_group.attrs["num_steps"] = n_steps
    clip_group.attrs["dt"] = 0.02
    file.create_group("/" + clip_name + "/walkers")
    file.create_group("/" + clip_name + "/props")
    walker_group = file.create_group("/" + clip_name + "/walkers/walker_0")
    for k, v in mocap_features.items():
        if len(np.array(v).shape) == 3:
            v = np.transpose(v, (1, 2, 0))
            # print(v.shape)
            walker_group[k] = np.reshape(np.array(v), (-1, n_steps))
        elif len(np.array(v).shape) == 2:
            v = np.swapaxes(v, 0, 1)
            walker_group[k] = v
        else:
            walker_group[k] = v
