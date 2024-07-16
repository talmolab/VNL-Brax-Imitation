import jax
from jax import numpy as jp
from typing import Any

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax.base import Motion, Transform
from brax.mjx.pipeline import _reformat_contact
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


class HumanoidTracking(PipelineEnv):
    def __init__(
        self,
        params,
        mjcf_path: str = "./assets/humanoid.xml",
        iterations: int = 6,
        ls_iterations: int = 6,
        solver: str = "cg",
        clip_length: int = 250,
        episode_length: int = 150,
        ref_traj_length: int = 5,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        termination_threshold: float = 10,
        body_error_multiplier: float = 1.0,
        scale_factor: float = 1,
        **kwargs,
    ):
        root = mjcf.from_path(mjcf_path)

        # Change actuators to torque (from positional)
        for actuator in root.find_all("actuator"):
            actuator.gainprm = [actuator.forcerange[1]]
            del actuator.biastype
            del actuator.biasprm
        root = mjcf.from_path(mjcf_path)

        # TODO: replace this rescale with jax version (from james cotton BodyModels)
        rescale.rescale_subtree(
            root,
            scale_factor,
            scale_factor,
        )
        
        mj_model = mjcf.Physics.from_mjcf_model(root).model.ptr

        mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.jacobian = 0  # Dense is faster on GPU

        sys = mjcf_brax.load_model(mj_model)

        physics_steps_per_control_step = 5

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._termination_threshold = termination_threshold
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._body_error_multiplier = body_error_multiplier
        self._clip_length = clip_length
        self._episode_length = episode_length
        self._ref_traj_length = ref_traj_length
        self._body_error_multiplier = body_error_multiplier

        with open(params["clip_path"], "rb") as f:
            self._ref_traj = pickle.load(f)

        if self._episode_length > self._clip_length:
            raise ValueError("episode_length cannot be greater than clip_length!")

    def reset(self, rng) -> State:
        """
        Resets the environment to an initial state.
        TODO: add a small amt of noise (qpos + epsilon) for randomization purposes
        """
        rng, subkey = jax.random.split(rng)

        start_frame = jax.random.randint(
            subkey,
            (),
            0,
            self._clip_length - self._episode_length - self._ref_traj_length,
        )

        qpos = jp.hstack(
            [
                self._ref_traj.position[start_frame, :],
                self._ref_traj.quaternion[start_frame, :],
                self._ref_traj.joints[start_frame, :],
            ]
        )
        qvel = jp.hstack(
            [
                self._ref_traj.velocity[start_frame, :],
                self._ref_traj.angular_velocity[start_frame, :],
                self._ref_traj.joints_velocity[start_frame, :],
            ]
        )
        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)
        traj = self._get_traj(data, start_frame)

        info = {
            "cur_frame": start_frame,
            "traj": traj,
        }

        reward, done, zero = jp.zeros(3)
        metrics = {
            "rcom": zero,
            "rvel": zero,
            "rtrunk": zero,
            "rquat": zero,
            "ract": zero,
            "termination_error": zero,
        }

        state = State(data, obs, reward, done, metrics, info)
        termination_error = self._calculate_termination(state)
        info["termination_error"] = termination_error
        # if termination_error > 1e-1:
        #   raise ValueError(('The termination exceeds 1e-2 at initialization. '
        #                     'This is likely due to a proto/walker mismatch.'))
        state = state.replace(info=info)

        return state

    def reset_to_frame(self, start_frame) -> State:
        """
        Resets the environment to the initial frame
        """
        qpos = jp.hstack(
            [
                self._ref_traj.position[start_frame, :],
                self._ref_traj.quaternion[start_frame, :],
                self._ref_traj.joints[start_frame, :],
            ]
        )
        qvel = jp.hstack(
            [
                self._ref_traj.velocity[start_frame, :],
                self._ref_traj.angular_velocity[start_frame, :],
                self._ref_traj.joints_velocity[start_frame, :],
            ]
        )
        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)
        traj = self._get_traj(data, start_frame)
        info = {
            "cur_frame": start_frame,
            "traj": traj,
        }

        reward, done, zero = jp.zeros(3)
        metrics = {
            "rcom": zero,
            "rvel": zero,
            "rtrunk": zero,
            "rquat": zero,
            "ract": zero,
            "termination_error": zero,
        }

        state = State(data, obs, reward, done, metrics, info)
        termination_error = self._calculate_termination(state)
        info["termination_error"] = termination_error
        # if termination_error > 1e-1:
        #   raise ValueError(('The termination exceeds 1e-2 at initialization. '
        #                     'This is likely due to a proto/walker mismatch.'))
        state = state.replace(info=info)

        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        info = state.info.copy()
        info["cur_frame"] += 1

        obs = self._get_obs(data)
        traj = self._get_traj(data, info["cur_frame"])

        rcom, rvel, rtrunk, rquat, ract, is_healthy = self._calculate_reward(
            state, action
        )
        done = jp.where((rtrunk < 0.5), jp.array(1, float), jp.array(0, float))
        rcom *= 0.01
        rvel *= 0.01
        rtrunk *= 0.01
        rquat *= 0.01
        ract *= 0.0001
        total_reward = rcom + rvel + rtrunk + rquat + ract
        
        # increment frame tracker and up
        # date termination error
        info["termination_error"] = rtrunk
        info["traj"] = traj

        done = jp.max(jp.array([1.0 - is_healthy, done]))
        reward = jp.nan_to_num(total_reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        done = jp.where(num_nans > 0, 1.0, done)

        state.metrics.update(
            rcom=rcom,
            rvel=rvel,
            rquat=rquat,
            ract=ract,
            rtrunk=rtrunk,
            termination_error=rtrunk,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
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
        data_c = state.pipeline_state

        target_joints = self._ref_traj.joints[state.info["cur_frame"], :]
        error_joints = jp.linalg.norm((target_joints - data_c.qpos[7:]), ord=1)

        target_bodies = self._ref_traj.body_positions[state.info["cur_frame"], :]
        error_bodies = jp.linalg.norm(
            (target_bodies - data_c.xpos), ord=1
        )
        error = 0.5 * self._body_error_multiplier * error_bodies + 0.5 * error_joints
        termination_error = 1 - (
            error / self._termination_threshold
        )  # low threshold, easier to terminate, more sensitive

        is_healthy = jp.where(data_c.q[2] < self._healthy_z_range[0], 0.0, 1.0)
        is_healthy = jp.where(data_c.q[2] > self._healthy_z_range[1], 0.0, is_healthy)

        return termination_error, is_healthy

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
        com_ref = self._ref_traj.center_of_mass[state.info["cur_frame"], :]
        rcom = jp.exp(-100 * (jp.linalg.norm(com_c - (com_ref))))

        # joint angle velocity
        qvel_c = data_c.qvel
        qvel_ref = jp.hstack(
            [
                self._ref_traj.velocity[state.info["cur_frame"], :],
                self._ref_traj.angular_velocity[state.info["cur_frame"], :],
                self._ref_traj.joints_velocity[state.info["cur_frame"], :],
            ]
        )
        rvel = jp.exp(-0.1 * (jp.linalg.norm(qvel_c - (qvel_ref))))

        # rtrunk = termination error
        rtrunk = self._calculate_termination(state)

        quat_c = data_c.qpos[3:7]
        quat_ref = self._ref_traj.quaternion[state.info["cur_frame"], :]
        
        # use bounded_quat_dist from dmcontrol
        rquat = jp.exp(-2 * (jp.linalg.norm(self._bounded_quat_dist(quat_c, quat_ref))))

        # control force from actions
        ract = -0.015 * jp.mean(jp.square(data_c.qfrc_actuator))

        is_healthy = jp.where(data_c.q[2] < self._healthy_z_range[0], 0.0, 1.0)
        is_healthy = jp.where(data_c.q[2] > self._healthy_z_range[1], 0.0, is_healthy)
        return rcom, rvel, rtrunk, rquat, ract, is_healthy

    def _get_traj(self, data: mjx.Data, cur_frame: int) -> jp.ndarray:
        """
        Gets reference trajectory obs along with env state obs
        """

        # Get the relevant slice of the ref_traj
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    cur_frame + 1,
                    self._ref_traj_length,
                )
            return jp.array([])

        ref_traj = jax.tree_util.tree_map(f, self._ref_traj)

        reference_rel_bodies_pos_local = self.get_reference_rel_bodies_pos_local(
            data, ref_traj
        )
        reference_rel_bodies_pos_global = self.get_reference_rel_bodies_pos_global(
            data, ref_traj
        )
        reference_rel_root_pos_local = self.get_reference_rel_root_pos_local(
            data, ref_traj
        )
        reference_rel_joints = self.get_reference_rel_joints(data, ref_traj)

        return jp.concatenate(
            [
                reference_rel_bodies_pos_local,
                reference_rel_bodies_pos_global,
                reference_rel_root_pos_local,
                reference_rel_joints,
            ]
        )

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        """
        Gets reference trajectory obs along with env state obs
        """
        end_effectors = data.xpos.flatten()

        return jp.concatenate(
            [
                data.qpos,
                data.qvel,
                data.qfrc_actuator,
                end_effectors,
            ]
        )
    
    def global_vector_to_local_frame(self, data, vec_in_world_frame):
        """Linearly transforms a world-frame vector into entity's local frame.

        Note that this function does not perform an affine transformation of the
        vector. In other words, the input vector is assumed to be specified with
        respect to the same origin as this entity's local frame. This function
        can also be applied to matrices whose innermost dimensions are either 2 or
        3. In this case, a matrix with the same leading dimensions is returned
        where the innermost vectors are replaced by their values computed in the
        local frame.

        Returns the resulting vector, converting to ego-centric frame
        """
        # TODO: confirm index
        xmat = jp.reshape(data.xmat[1], (3, 3))
        # The ordering of the np.dot is such that the transformation holds for any
        # matrix whose final dimensions are (2,) or (3,).

        # Each element in xmat is a 3x3 matrix that describes the rotation of a body relative to the global coordinate frame, so
        # use rotation matrix to dot the vectors in the world frame, transform basis
        if vec_in_world_frame.shape[-1] == 2:
            return jp.dot(vec_in_world_frame, xmat[:2, :2])
        elif vec_in_world_frame.shape[-1] == 3:
            return jp.dot(vec_in_world_frame, xmat)
        else:
            raise ValueError(
                "`vec_in_world_frame` should have shape with final "
                "dimension 2 or 3: got {}".format(vec_in_world_frame.shape)
            )

    def get_reference_rel_bodies_pos_local(self, data, ref_traj):
        """Observation of the reference bodies relative to walker in local frame."""
        xpos_broadcast = jp.broadcast_to(
            data.xpos, ref_traj.body_positions.shape
        )
        obs = self.global_vector_to_local_frame(
            data, ref_traj.body_positions - xpos_broadcast
        )
        return jp.concatenate([o.flatten() for o in obs])

    def get_reference_rel_bodies_pos_global(self, data, ref_traj):
        """Observation of the reference bodies relative to walker, global frame directly"""
        xpos_broadcast = jp.broadcast_to(
            data.xpos, ref_traj.body_positions.shape
        )
        diff = ref_traj.body_positions - xpos_broadcast

        return diff.flatten()

    def get_reference_rel_root_pos_local(self, data, ref_traj):
        """Reference position relative to current root position in root frame."""
        diff = ref_traj.position - data.qpos[:3]
        obs = self.global_vector_to_local_frame(data, diff)
        return jp.concatenate([o.flatten() for o in obs])

    def get_reference_rel_joints(self, data, ref_traj):
        """Observation of the reference joints relative to walker."""
        diff = (ref_traj.joints - data.qpos[7:])
        return diff.flatten()


    def _bounded_quat_dist(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes a quaternion distance limiting the difference to a max of pi/2.

        This function supports an arbitrary number of batch dimensions, B.

        Args:
          source: a quaternion, shape (B, 4).
          target: another quaternion, shape (B, 4).

        Returns:
          Quaternion distance, shape (B, 1).
        """
        source /= jp.linalg.norm(source, axis=-1, keepdims=True)
        target /= jp.linalg.norm(target, axis=-1, keepdims=True)
        # "Distance" in interval [-1, 1].
        dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
        # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
        dist = jp.minimum(1.0, dist)
        # Divide by 2 and add an axis to ensure consistency with expected return
        # shape and magnitude.
        return 0.5 * jp.arccos(dist)[..., np.newaxis]

class HumanoidStanding(PipelineEnv):
    def __init__(
        self,
        params,
        forward_reward_weight=0.0,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        sys = mjcf_brax.load_model(
            mujoco.MjModel.from_xml_path("./assets/humanoid.xml")
        )
        sys = sys.tree_replace(
            {
                "opt.solver": {
                    "cg": mujoco.mjtSolver.mjSOL_CG,
                    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
                }[params["solver"].lower()],
                "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                "opt.iterations": params["iterations"],
                "opt.ls_iterations": params["ls_iterations"],
                "opt.jacobian": 0,  # Dense matrix
            }
        )

        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        return State(data, obs, reward, done, metrics)

    def reset_to_frame(self, start_frame):
        # TODO: make the rng stuff in reset a wrapper? and reset_to_frame is default
        return self.reset(jax.random.PRNGKey(0))

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

        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        done = jp.where(num_nans > 0, 1.0, done)

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

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )
