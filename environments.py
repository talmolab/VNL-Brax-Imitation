import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

import mujoco
from mujoco import mjx
import numpy as np
import h5py
import os
from mujoco.mjx._src.dataclasses import PyTreeNode
from walker import Rat

_XML_PATH = "assets/rodent.xml"

# 13 features
# TODO: currently stores np arrays
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
  walker = Rat(foot_mods=False)
  rescale.rescale_subtree(
      walker.mjcf_model,
      params["scale_factor"],
      params["scale_factor"],
  )
  physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)

  # get mjmodel from physics and set up solver configs
  mj_model = physics.model.ptr
  
  mj_model.opt.solver = {
    'cg': mujoco.mjtSolver.mjSOL_CG,
    'newton': mujoco.mjtSolver.mjSOL_NEWTON,
  }[params["solver"].lower()]
  mj_model.opt.iterations = params["iterations"]
  mj_model.opt.ls_iterations = params["ls_iterations"]
  mj_model.opt.jacobian = 0 # dense
  
  # gets the indices for end effectors: [11, 15, 59, 64]
  axis = physics.named.model.body_pos._axes[0]
  # app_idx = {key: int(axis.convert_key_item(key)) for key in utils.params["KEYPOINT_MODEL_PAIRS"].keys()}
  end_eff_idx = [int(axis.convert_key_item(key)) 
                 for key in params['end_eff_names']]
  walker_bodies = walker.mocap_tracking_bodies
  # body_pos_idx
  walker_bodies_names = [bdy.name for bdy in walker_bodies]
  body_idxs = jp.array(
    [walker_bodies_names.index(bdy) for bdy in walker_bodies_names]
  )
  
  return mj_model, end_eff_idx, body_idxs
  
  
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
    mj_model, self._end_eff_idx, self.body_idxs = env_setup(params)

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
    
    # do i need to subtract another 1? getobs gives the next n frames
    start_frame = jax.random.randint(
      subkey, (), 0, 
      self._clip_length - self._episode_length - self._ref_traj_length
    )

    # qpos = position + quaternion + joints
    # pos = self._ref_traj.position[:, start_frame]
    # quat = self._ref_traj.quaternion[:, start_frame]
    # joints = self._ref_traj.joints[:, start_frame]
    # qpos = jp.concatenate((pos, quat, joints))
    qpos = jp.hstack([
      self._ref_traj.position[:, start_frame],
      self._ref_traj.quaternion[:, start_frame],
      self._ref_traj.joints[:, start_frame],
    ])
    qvel = jp.hstack([
      self._ref_traj.velocity[:, start_frame],
      self._ref_traj.angular_velocity[:, start_frame],
      self._ref_traj.joints_velocity[:, start_frame],
    ])
    data = self.pipeline_init(qpos, qvel) # jp.zeros(self.sys.nv) 

    info = {
      "start_frame": start_frame,
      "next_frame": start_frame + 1
    }
    obs = self._get_obs(data, jp.zeros(self.sys.nu), info)
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

    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    # increment frame tracker
    state.info['next_frame'] += 1
    
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
      self, data: mjx.Data, action: jp.ndarray, info
  ) -> jp.ndarray:
    """
      Gets reference trajectory obs along with env state obs 
    """

    # This should get the relevant slice of the ref_traj, and flatten/concatenate into a 1d vector
    # Then transform it before returning with the rest of the obs
    ref_traj = self._ref_traj.qpos[info['next_frame']:info['next_frame'] + self._ref_traj_length]
    ref_traj = self.get_reference_rel_bodies_pos_local(data, ref_traj)
    
    # TODO: end effectors pos and appendages pos are two different features?
    end_effectors = data.xpos[self._end_eff_idx] 
    return jp.concatenate([
      # put the traj obs first
        data.qpos, 
        data.qvel, 
        data.qfrc_actuator, # Actuator force <==> joint torque sensor?
        end_effectors,
    ])
  
  def global_vector_to_local_frame(self, mjxData, vec_in_world_frame):
    """Linearly transforms a world-frame vector into entity's local frame.

    Note that this function does not perform an affine transformation of the
    vector. In other words, the input vector is assumed to be specified with
    respect to the same origin as this entity's local frame. This function
    can also be applied to matrices whose innermost dimensions are either 2 or
    3. In this case, a matrix with the same leading dimensions is returned
    where the innermost vectors are replaced by their values computed in the
    local frame.
    
    Returns the resulting vector
    """
    xmat = jp.reshape(mjxData.xmat, (3, 3))
    # The ordering of the np.dot is such that the transformation holds for any
    # matrix whose final dimensions are (2,) or (3,).
    if vec_in_world_frame.shape[-1] == 2:
      return jp.dot(vec_in_world_frame, xmat[:2, :2])
    elif vec_in_world_frame.shape[-1] == 3:
      return jp.dot(vec_in_world_frame, xmat)
    else:
      raise ValueError('`vec_in_world_frame` should have shape with final '
                       'dimension 2 or 3: got {}'.format(
                           vec_in_world_frame.shape))

  def get_reference_rel_bodies_pos_local(self, data, clip_reference_features, frame):
    """Observation of the reference bodies relative to walker in local frame."""
    
    # self._walker_features['body_positions'] is the equivalent of 
    # the ref traj 'body_positions' feature but calculated for the current walker state
    # TODO: self._body_postioins_idx does not exist yet (how is it different from body_idxs?)
    time_steps = frame + jp.arange(self._ref_traj_length)
    obs = self.global_vector_to_local_frame(
        data, (clip_reference_features['body_positions'][time_steps] -
                data[self._body_positions_idx])[:, self._body_idxs])
    return jp.concatenate([o.flatten() for o in obs])
    