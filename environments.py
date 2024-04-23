import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

import mujoco
from mujoco import mjx

import numpy as np

import os

_XML_PATH = "rodent.xml"


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
        params["SCALE_FACTOR"],
        params["SCALE_FACTOR"],
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
  
  
class RodentTrackClip(PipelineEnv):

  def __init__(
      self,
      params,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.01, 0.5),
      reset_noise_scale=1e-2,
      solver="cg",
      iterations: int = 6,
      ls_iterations: int = 3,
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
    
  def reset(self, rng) -> State:
    """
    Resets the environment to an initial state.
    TODO: Must reset this to the start of a trajectory (set the appropriate qpos)
    Can still add a small amt of noise (qpos + epsilon) for randomization purposes
    """
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel)

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
    return State(data, obs, reward, done, metrics)

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
    ref_traj = transform_to_relative(ref_traj)
    return jp.concatenate([
        data.qpos, 
        data.qvel, 
        data.qfrc_actuator, # Actuator force <==> joint torque sensor?
        data.geom_xpos, # 
    ])
    