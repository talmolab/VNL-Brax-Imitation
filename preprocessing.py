import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx

from typing import Text, Tuple
import pickle

from preprocessing_utils import compute_velocity_from_kinematics

"""
This file is takes stac'd data (essentially a list of qposes) 
and processes it to get a full set of trajectory data.
A shortened list of trajectory data for a first iteration:
- joint positions (qpos)
- body positions (xpos)
- body quaternions (xquat)
- center of mass (subtree_com)
- velocity
- angular velocity
- joints velocity
"""

# TODO: make a MocapFeatures jax dataclass so it's not just a dict? 
# or do that when loading the data for training and not at this stage

def get_mocap_features(
    stac_path: Text,
    save_file: Text,
    start_step: int = 0,
    clip_length: int = 2500,
    dt: float = 0.02,
    max_qvel: float = 20.0,
    adjust_z_offset: float = 0.0,
    verbatim: bool = False,
    ref_steps: Tuple = (1, 2, 3, 4, 5)
    ):

    with open(stac_path, "rb") as file:
        d = pickle.load(file)
        mocap_qpos = jnp.array(d["qpos"])
    
    mocap_features = dict()
    
    z_offset = 0
    
    # What is this feet_height thing...
    # if adjust_z_offset:
    #         z_offset = feet_height[:10].mean() - 0.006        
            
    mocap_qpos[:, 2] -= z_offset

    mocap_qvel = compute_velocity_from_kinematics(mocap_qpos, dt)
    vels = mocap_qvel[:, 6:]
    clipped_vels = jnp.clip(vels, -max_qvel, max_qvel)
    indexes = jnp.where(vels != clipped_vels)
    if verbatim and indexes[0].size != 0:
        for i, j in zip(*indexes):
            if jnp.abs(vels[i, j] - clipped_vels[i, j]) >= 0.1:
                print(
                    "Step {} velocity of {} clipped from {} to {}.".format(
                        i, joint_names[j], vels[i, j], clipped_vels[i, j] # can get joint names logic from stac-mjx if needed
                    )
                )
    mocap_qvel[:, 6:] = clipped_vels
    mocap_features["velocity"] = mocap_qvel[:, :3]
    mocap_features["angular_velocity"] = mocap_qvel[:, 3:6]
    mocap_features["joints_velocity"] = mocap_qvel[:, 6:]
        

