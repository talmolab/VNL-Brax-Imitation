import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

import mujoco
from mujoco import mjx
from flax import struct
import numpy as np
from dataclasses import dataclass
import h5py
import os
from mujoco.mjx._src.dataclasses import PyTreeNode

_XML_PATH = "assets/rodent.xml"

# 13 features
# @struct.dataclass
class ReferenceClip(PyTreeNode):
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
  
  
def unpack_clip(file_path):
  """Creates a ReferenceClip dataclass with the clip data.
  Returns: 
      ReferenceClip: containing the entire clip's worth of trajectory data
  """
  with h5py.File(file_path, "r") as f:
    data_group = f['clip_0']['walkers']['walker_0']
    clip = ReferenceClip(
      np.array(data_group['angular_velocity']),
      np.array(data_group['appendages']),
      np.array(data_group['body_positions']),
      np.array(data_group['body_quaternions']),
      np.array(data_group['center_of_mass']),
      np.array(data_group['end_effectors']),
      np.array(data_group['joints']),
      np.array(data_group['joints_velocity']),
      np.array(data_group['markers']),
      np.array(data_group['position']),
      np.array(data_group['quaternion']),
      np.array(data_group['scaling']),
      np.array(data_group['velocity']),
    )
  return clip

def env_setup(params):
  """sets up the mj_model on intialization with help from dmcontrol
  rescales model, gets end effector indices, and more?

  Args:
      params (_type_): _description_

  Returns:
      _type_: _description_
  """
  root = mjcf.from_path(_XML_PATH)
  
  rescale.rescale_subtree(
      root,
      params["scale_factor"],
      params["scale_factor"],
  )
  physics = mjcf.Physics.from_mjcf_model(root)

  # get mjmodel from physics and set up solver configs
  mj_model = physics.model.ptr
  
  mj_model.opt.solver = {
    'cg': mujoco.mjtSolver.mjSOL_CG,
    'newton': mujoco.mjtSolver.mjSOL_NEWTON,
  }[params["solver"].lower()]
  mj_model.opt.iterations = params["iterations"]
  mj_model.opt.ls_iterations = params["ls_iterations"]
  mj_model.opt.jacobian = 0 # dense
  
  return mj_model
  
  
class RodentSingleClipTrack(PipelineEnv):

  def __init__(
      self,
      params,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.01, 0.5),
      reset_noise_scale=1e-2,
      clip_length: int=250,
      episode_length: int=150,
      ref_traj_length: int=5,
      **kwargs,
  ):
    mj_model = env_setup(params)

    sys = mjcf_brax.load_model(mj_model)

    physics_steps_per_control_step = 5
    
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step
    )
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._clip_length = clip_length
    self._episode_length = episode_length
    self._ref_traj_length = ref_traj_length
    self._ref_traj = unpack_clip(params["clip_path"])
    if self._episode_length > self._clip_length:
      raise ValueError("episode_length cannot be greater than clip_length!")
    
  def reset(self, rng) -> State:
    """
    Resets the environment to an initial state.
    TODO: Must reset this to the start of a trajectory (set the appropriate qpos)
    Can still add a small amt of noise (qpos + epsilon) for randomization purposes
    """
    rng, subkey = jax.random.split(rng)
    
    start_frame = jax.random.randint(subkey,(), 0, self._clip_length - self._episode_length)

    # qpos = position + quaternion + joints
    pos = self._ref_traj.position[:, start_frame]
    quat = self._ref_traj.quaternion[:, start_frame]
    joints =self._ref_traj.joints[:, start_frame]
    print(pos.shape, quat.shape, joints.shape)
    qpos = jp.concatenate((pos, quat, joints))
    print(qpos.shape)
    
    data = self.pipeline_init(qpos, jp.zeros(self.sys.nv))

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    info = {
      "start_frame": start_frame
    }
    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    com_before = data0.subtree_com[1]
    com_after = data.subtree_com[1]
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(data, action)
    reward = forward_reward + healthy_reward - ctrl_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
    # TODO update these metrics (new reward components)
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )


  def _calculate_termination(self, state, ref) -> bool:
    """calculates whether the termination condition is met

    Args:
        state (_type_): _description_
        ref (_type_): reference trajectory

    Returns:
        bool: _description_
    """
    return 
    
  def _calculate_reward(self, state):
    """calculates the tracking reward:
    (insert description of each of the terms)

    Args:
        state (_type_): _description_
    """
    
    total_reward = rcom + rvel + rapp + rquat + ract
    return total_reward

  def _get_obs(
      self, data: mjx.Data, action: jp.ndarray
  ) -> jp.ndarray:
    """
      Gets reference trajectory obs along with env state obs 
    """
    # I don't think the order matters as long as it's consistent?
    # return the reference traj concatenated with the state obs. reference traj goes in the encoder
    # and the rest of the obs go straight to the decoder
    ref_traj = self.full_ref_traj.qpos[i:i+5]
    ref_traj = transform_to_relative(data, ref_traj)
    return jp.concatenate([
        data.qpos, 
        data.qvel, 
        data.qfrc_actuator, # Actuator force <==> joint torque sensor?
        data.geom_xpos, # 
    ])
    