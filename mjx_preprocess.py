"""Preprocess mocap data for mjx"""

import jax
from jax import jit
from jax import numpy as jnp

import numpy as np

import mujoco
from mujoco import mjx
from mujoco.mjx._src import smooth

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

from typing import Text, Tuple
import pickle


@jit
def kinematics(mjx_model: mjx.Model, mjx_data: mjx.Data):
    """jit compiled forward kinematics

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):

    Returns:
        mjx.Data: resulting mjx Data
    """
    return smooth.kinematics(mjx_model, mjx_data)


@jit
def com_pos(mjx_model: mjx.Model, mjx_data: mjx.Data):
    """jit compiled com_pos calculation

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):

    Returns:
        mjx.Data: resulting mjx Data
    """
    return smooth.com_pos(mjx_model, mjx_data)


@jit
def set_qpos(mjx_model: mjx.Model, mjx_data: mjx.Data, qpos: jnp.Array) -> mjx.Data:
    """Sets the qpos and performs forward kinematics (zeros for qvel)

    Args:
        mjx_model (mjx.Model): _description_
        mjx_data (mjx.Data): _description_
        qpos (jnp.Array): _description_

    Returns:
        mjx.Data: _description_
    """
    qvel = jnp.zeros((mjx_model.nv,))
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = kinematics(mjx_model, mjx_data)
    return mjx_data


def process(
    stac_path: Text,
    save_file: Text,
    scale_factor: float = 0.9,
    start_step: int = 0,
    clip_length: int = 250,
    max_qvel: float = 20.0,
    dt: float = 0.02,
    ref_steps: Tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    """Process a set of joint angles into the features that
       the referenced trajectory is composed of. Unlike the original,
       this function will process and save only one clip.
       Once this is all ported to jax, it can be vmapped to parallelize the preprocessing

    Args:
        stac_path (Text): _description_
        save_file (Text): _description_
        scale_factor (float, optional): _description_. Defaults to 0.9.
        start_step (int, optional): _description_. Defaults to 0.
        clip_length (int, optional): _description_. Defaults to 250.
        max_qvel (float, optional): _description_. Defaults to 20.0.
        dt (float, optional): _description_. Defaults to 0.02.
        ref_steps (Tuple, optional): _description_. Defaults to (1, 2, 3, 4, 5, 6, 7, 8, 9, 10).
    """
    with open(stac_path, "rb") as file:
        d = pickle.load(file)
        mocap_qpos = np.array(d["qpos"])

    # Load rodent mjcf and rescale, then get the mj_model from that.
    # TODO: make this all work in mjx? james cotton did rescaling with mjx model:
    # https://github.com/peabody124/BodyModels/blob/f6ef1be5c5d4b7e51028adfc51125e510c13bcc2/body_models/biomechanics_mjx/forward_kinematics.py#L92
    # TODO: Set this up outside of this function as it only needs to be done once anyway

    root = mjcf.from_path("./assets/rodent.xml")
    rescale.rescale_subtree(
        root,
        scale_factor,
        scale_factor,
    )
    mj_model = mjcf.Physics.from_mjcf_model(root).model.ptr
    mj_data = mujoco.MjData(mj_model)

    # Place into GPU
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    return
