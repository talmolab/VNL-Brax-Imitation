import jax
from jax import numpy as jp
from typing import Any

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax.base import Motion, Transform
from brax.mjx.pipeline import _reformat_contact

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

import mujoco
from mujoco import mjx
import numpy as np
import h5py
import os
from mujoco.mjx._src.dataclasses import PyTreeNode
from walker import Rat
import pickle

from mocap_preprocess import ReferenceClip

_XML_PATH = "assets/rodent.xml"
  
  
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
  mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  
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
  end_eff_idx = jp.array([int(axis.convert_key_item(key)) 
                 for key in params['end_eff_names']])
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
      termination_threshold: float=.3,
      body_error_multiplier: float=1.0,
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
    # self._ref_traj = unpack_clip(params["clip_path"])
    self._termination_threshold = termination_threshold
    self._body_error_multiplier = body_error_multiplier


    with open(params["clip_path"], 'rb') as f:
      self._ref_traj = pickle.load(f)
      
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
      self._ref_traj.position[start_frame, :],
      self._ref_traj.quaternion[start_frame, :],
      self._ref_traj.joints[start_frame, :],
    ])
    qvel = jp.hstack([
      self._ref_traj.velocity[start_frame, :],
      self._ref_traj.angular_velocity[start_frame, :],
      self._ref_traj.joints_velocity[start_frame, :],
    ])
    data = self.pipeline_init(qpos, qvel) # jp.zeros(self.sys.nv) 

    info = {
      "cur_frame": start_frame,
    }
    obs = self._get_obs(data, jp.zeros(self.sys.nu), info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'rcom': zero,
        'rvel': zero,
        'rapp': zero,
        'rquat': zero,
        'ract': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }

    state = State(data, obs, reward, done, metrics, info)
    termination_error = self._calculate_termination(state)
    info['termination_error'] = termination_error
    # if termination_error > 3e-2:
    #   raise ValueError(('The termination exceeds 1e-2 at initialization. '
    #                     'This is likely due to a proto/walker mismatch.'))
    state = state.replace(info=info)
    
    return state

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    com_before = data0.subtree_com[1]
    com_after = data.subtree_com[1]
    velocity = (com_after - com_before) / self.dt

    obs = self._get_obs(data, action, state.info)
    rcom, rvel, rquat, ract, rapp = self._calculate_reward(state, action)
    total_reward = rcom + rvel + rapp + rquat + ract
    
    termination_error = self._calculate_termination(state)
    
    # increment frame tracker and update termination error
    info = state.info.copy()
    info['termination_error'] = termination_error
    info['cur_frame'] += 1

    done = termination_error > self._termination_threshold

    state.metrics.update(
        rcom=rcom,
        rvel=rvel,
        rapp=rapp,
        rquat=rquat,
        ract=ract,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )
    
    return state.replace(
        pipeline_state=data, obs=obs, reward=total_reward, done=done, info=info
    )


  def _calculate_termination(self, state) -> float:
    """
    calculates whether the termination condition is met
    Args:
        state (_type_): _description_
        ref (_type_): reference trajectory
    Returns:
        bool: _description_
    """
    # qpos_c = data_c.qpos
    # qpos_ref = jp.hstack([
    #   self._ref_traj.position[state.info['cur_frame'], :],
    #   self._ref_traj.quaternion[state.info['cur_frame'], :],
    #   self._ref_traj.joints[state.info['cur_frame'], :]
    # ])
    # bpos_c = data_c.xpos[self.body_idxs] # (18 (spots) x 3 dimension(x,y,z))
    # bpos_ref = self._ref_traj.body_position[state.info['cur_frame'], :] # (18 (spots) x 3 dimension (x,y,z))
    # return 1 - (1/0.3) * ((jp.linalg.norm(bpos_c - (bpos_ref))) + 
    #                   (jp.linalg.norm(qpos_c - (qpos_ref)))) < 0

    data_c = state.pipeline_state
    
    target_joints = self._ref_traj.joints[state.info['cur_frame'], :]
    error_joints = jp.mean(jp.abs(target_joints - data_c.qpos[7:]))
    target_bodies = self._ref_traj.body_positions[state.info['cur_frame'], :]
    error_bodies = jp.mean(jp.abs((target_bodies - data_c.xpos[self.body_idxs])))

    termination_error = (0.5 * self._body_error_multiplier * error_bodies + 0.5 * error_joints)
    
    return termination_error
    
  def _calculate_reward(self, state, action):
    """
    calculates the tracking reward:
    1. rcom: comparing center of mass
    2. rvel: comparing joint angle velcoity
    3. rquat: comprae joint angle position
    4. ract: compare control force
    5. rapp: compare end effector appendage positions
    Args:
        state (_type_): _description_
    """
    data_c = state.pipeline_state

    # location using com (dim=3)
    com_c = data_c.subtree_com[1]
    com_ref = self._ref_traj.center_of_mass[state.info['cur_frame'], :]
    rcom = jp.exp(-100 * (jp.linalg.norm(com_c - (com_ref))**2))

    # joint angle velocity
    qvel_c = data_c.qvel
    qvel_ref = jp.hstack([
      self._ref_traj.velocity[state.info['cur_frame'], :],
      self._ref_traj.angular_velocity[state.info['cur_frame'], :],
      self._ref_traj.joints_velocity[state.info['cur_frame'], :],
    ])
    rvel = jp.exp(-0.1 * (jp.linalg.norm(qvel_c - (qvel_ref))**2))

    # joint angle posiotion
    qpos_c = data_c.qpos
    qpos_ref = jp.hstack([
      self._ref_traj.position[state.info['cur_frame'], :],
      self._ref_traj.quaternion[state.info['cur_frame'], :],
      self._ref_traj.joints[state.info['cur_frame'], :],
    ])
    rquat = jp.exp(-2 * (jp.linalg.norm(qpos_c - (qpos_ref))**2))

    # control force from actions
    ract = -0.015 * jp.sum(jp.square(action)) / len(action)
   
    # end effector positions
    app_c = data_c.xpos[jp.array(self._end_eff_idx)].flatten()
    app_ref = self._ref_traj.end_effectors[state.info['cur_frame'], :].flatten()

    rapp = jp.exp(-400 * (jp.linalg.norm(app_c - (app_ref))**2))
    return rcom, rvel, rquat, ract, rapp
  

  def _get_obs(
      self, data: mjx.Data, action: jp.ndarray, info
  ) -> jp.ndarray:
    """
      Gets reference trajectory obs along with env state obs 
    """
    # This should get the relevant slice of the ref_traj, and flatten/concatenate into a 1d vector
    # Then transform it before returning with the rest of the obs
    
    # info is currently a global variable
    # ref_traj = self._ref_traj.body_positions[:, info['next_frame']:info['next_frame'] + self._ref_traj_length]
    # ref_traj = jp.hstack(ref_traj)
    
    # slicing function apply outside of data class
    def f(x):
      if len(x.shape) != 1:
        return jax.lax.dynamic_slice_in_dim(
          x, 
          info['cur_frame'] + 1, 
          self._ref_traj_length, 
        )
      return jp.array([])
    
    ref_traj = jax.tree_util.tree_map(f, self._ref_traj)
    # ref_traj_flat = ref_traj.flatten_attributes()
    
    # now being a local variable
    reference_rel_bodies_pos_local = self.get_reference_rel_bodies_pos_local(data, ref_traj, info['cur_frame'] + 1)
    
    # TODO: end effectors pos and appendages pos are two different features?
    end_effectors = data.xpos[self._end_eff_idx].flatten()

    return jp.concatenate([
      # put the traj obs first
        reference_rel_bodies_pos_local,
        data.qpos, 
        data.qvel, 
        data.qfrc_actuator, # Actuator force <==> joint torque sensor?
        end_effectors,
    ])
  
  def global_vector_to_local_frame(self, data, vec_in_world_frame):
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
    # [0] is the root_body index
    xmat = jp.reshape(data.xmat[0], (3, 3))
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
    
  def get_reference_rel_bodies_pos_local(self, data, ref_traj, frame):
    """Observation of the reference bodies relative to walker in local frame."""
    
    # self._walker_features['body_positions'] is the equivalent of 
    # the ref traj 'body_positions' feature but calculated for the current walker state

    time_steps = frame + jp.arange(self._ref_traj_length)
    thing = (ref_traj.body_positions[time_steps] - data.xpos[self.body_idxs])
    # Still unsure why the slicing below is necessary but it seems this is what dm_control did..
    obs = self.global_vector_to_local_frame(
      data,
      thing[:, self.body_idxs]
    )
    
    return jp.concatenate([o.flatten() for o in obs])