import jax
from jax import numpy as jp
from typing import Any, Sequence

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

import mujoco
from mujoco import mjx
import numpy as np
import mocap_preprocess as mp


class RodentTracking(PipelineEnv):
    def __init__(
        self,
        params,
        healthy_z_range=(0.05, 0.5),
        reset_noise_scale=1e-3,
        clip_length: int = 250,
        sub_clip_length: int = 150,
        ref_traj_length: int = 5,
        termination_threshold: float = 5,
        body_error_multiplier: float = 1.0,
        explore_time: int = 20,
        **kwargs,
    ):
        root = mjcf.from_path("./assets/rodent.xml")

        # TODO: replace this rescale with jax version (from james cotton BodyModels)
        rescale.rescale_subtree(
            root,
            params["scale_factor"],
            params["scale_factor"],
        )
        mj_model = mjcf.Physics.from_mjcf_model(root).model.ptr

        mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[params["solver"].lower()]
        mj_model.opt.iterations = params["iterations"]
        mj_model.opt.ls_iterations = params["ls_iterations"]
        mj_model.opt.jacobian = 0  # dense

        self._end_eff_idx = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in params["end_eff_names"]
            ]
        )
        self._app_idx = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in params["appendage_names"]
            ]
        )
        self._com_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), params["center_of_mass"]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in params["walker_body_names"]
            ]
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in params["joint_names"]
            ]
        )

        sys = mjcf_brax.load_model(mj_model)

        physics_steps_per_control_step = 5

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._healthy_z_range = healthy_z_range
        self._explore_time = explore_time
        self._reset_noise_scale = reset_noise_scale
        self._termination_threshold = termination_threshold
        self._body_error_multiplier = body_error_multiplier
        self._clip_length = clip_length
        self._sub_clip_length = sub_clip_length
        self._ref_traj_length = ref_traj_length
        self._body_error_multiplier = body_error_multiplier
        self._ref_traj = params["reference_clip"]
        
        # initial fiteration
        filtered_bodies = self._ref_traj.body_positions[:, self._body_idxs]
        self._ref_traj = self._ref_traj.replace(body_positions=filtered_bodies)
        
        if self._sub_clip_length > self._clip_length:
            raise ValueError("episode_length cannot be greater than clip_length!")

    def reset(self, rng) -> State:
        """
        Resets the environment to an initial state.
        """
        start_frame = jax.random.randint(
            rng,
            (),
            0,
            self._clip_length - self._sub_clip_length - self._ref_traj_length,
            rng,
            (),
            0,
            self._clip_length - self._sub_clip_length - self._ref_traj_length,
        )

        old, rng = jax.random.split(rng)
        noise = self._reset_noise_scale * jax.random.normal(rng, shape=(self.sys.nq,))

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
        data = self.pipeline_init(qpos + noise, qvel)
        traj = self._get_traj(data, start_frame)

        info = {
            "cur_frame": start_frame,
            "traj": traj,
            "first_reset": 0,
        }
        obs = self._get_obs(data, jp.zeros(self.sys.nu), info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "rcom": zero,
            "rvel": zero,
            "rtrunk": zero,
            "rquat": zero,
            "ract": zero,
            "rapp": zero,
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
        traj = self._get_traj(data, start_frame)

        info = {
            "cur_frame": start_frame,
            "traj": traj,
            "first_reset": 0,
        }
        obs = self._get_obs(data, jp.zeros(self.sys.nu), info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "rcom": zero,
            "rvel": zero,
            "rtrunk": zero,
            "rquat": zero,
            "ract": zero,
            "rapp": zero,
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
        info["first_reset"] += 1

        obs = self._get_obs(data, action, state.info)
        traj = self._get_traj(data, info["cur_frame"])

        rcom, rvel, rtrunk, rquat, ract, rapp, is_healthy = self._calculate_reward(
            state, data
        )
        rcom *= 0.01
        rvel *= 0.01
        rapp *= 0.01
        rtrunk *= 0.01
        rquat *= 0.01
        ract *= 0.0001

        total_reward = rcom + rvel + rtrunk + rquat + ract + rapp

        # increment frame tracker and update termination error
        info["termination_error"] = rtrunk
        info["traj"] = traj

        sub_clip_healthy = jp.where(
            info["sub_clip_frame"] < self._sub_clip_length,
            jp.array(1, float),
            jp.array(0, float),
        )

        done = jp.where((rtrunk < 0), jp.array(1, float), jp.array(0, float))

        done = jp.where(
            (info["first_reset"] <= self._explore_time), jp.array(0, float), done
        )

        done = jp.max(jp.array([1.0 - is_healthy, done]))

        done = jp.max(jp.array([1.0 - sub_clip_healthy, done]))

        reward = jp.nan_to_num(total_reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        done = jp.where(num_nans > 0, 1.0, done)

        state.metrics.update(
            rcom=rcom,
            rvel=rvel,
            rapp=rapp,
            rquat=rquat,
            rtrunk=rtrunk,
            ract=ract,
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
            (target_bodies - data_c.xpos[self._body_idxs]), ord=1
        )
        error = 0.5 * self._body_error_multiplier * error_bodies + 0.5 * error_joints
        termination_error = 1 - (
            error / self._termination_threshold
        )  # low threshold, easier to terminate, more sensitive
        termination_error = 1 - (
            error / self._termination_threshold
        )  # low threshold, easier to terminate, more sensitive

        return termination_error

    def _calculate_reward(self, state, data_c):
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
        # location using com (dim=3)
        com_c = data_c.subtree_com[1]
        com_ref = self._ref_traj.body_positions[:, self._com_idx][
            state.info["cur_frame"], :
        ]
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
        rvel = jp.exp(-0.1 * (jp.linalg.norm(qvel_c - qvel_ref)))

        # rtrunk = termination error
        rtrunk = self._calculate_termination(state)

        quat_c = data_c.qpos[3:7]
        quat_ref = self._ref_traj.quaternion[state.info["cur_frame"], :]
        # use bounded_quat_dist from dmcontrol
        rquat = jp.exp(-2 * (jp.linalg.norm(self._bounded_quat_dist(quat_c, quat_ref))))

        # control force from actions
        ract = -0.015 * jp.mean(jp.square(data_c.qfrc_actuator))

        # end effector positions
        app_c = data_c.xpos[self._app_idx].flatten()
        app_ref = self._ref_traj.body_positions[:, self._app_idx][
            state.info["cur_frame"], :
        ].flatten()

        rapp = jp.exp(-400 * (jp.linalg.norm(app_c - app_ref)))

        is_healthy = jp.where(data_c.q[2] < self._healthy_z_range[0], 0.0, 1.0)
        is_healthy = jp.where(data_c.q[2] > self._healthy_z_range[1], 0.0, is_healthy)
        return rcom, rvel, rtrunk, rquat, ract, rapp, is_healthy

    def _get_obs(self, data: mjx.Data, action: jp.ndarray, info) -> jp.ndarray:
        """
        Get env state obs only
        """

        # Get the relevant slice of the ref_traj
        # TODO: can just use jax.lax.slice
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    info["cur_frame"] + 1,
                    self._ref_traj_length,
                )
            return jp.array([])

        # TODO: end effectors pos and appendages pos are two different features?
        end_effectors = data.xpos[self._end_eff_idx].flatten()

        return jp.concatenate(
            [
                data.qpos,
                data.qvel,
                data.qfrc_actuator,  # Actuator force <==> joint torque sensor?
                end_effectors,
            ]
        )


    def _get_traj(self, data: mjx.Data, cur_frame: int) -> jp.ndarray:
        """
        Gets reference trajectory obs for separate pathway, storage in the info section of state
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
        reference_appendages = self.get_reference_appendages_pos(ref_traj)
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
        reference_rel_joints = self.get_reference_rel_joints(data, ref_traj)

        return jp.concatenate(
            [
                reference_appendages,
                reference_rel_bodies_pos_local,
                reference_rel_bodies_pos_global,
                reference_rel_root_pos_local,
                reference_rel_joints,
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
            data.xpos[self._body_idxs], ref_traj.body_positions.shape
        )
        obs = self.global_vector_to_local_frame(
            data, ref_traj.body_positions - xpos_broadcast
        )  # [:, self._body_idxs]
        return jp.concatenate([o.flatten() for o in obs])

    def get_reference_rel_bodies_pos_global(self, data, ref_traj):
        """Observation of the reference bodies relative to walker, global frame directly"""
        xpos_broadcast = jp.broadcast_to(
            data.xpos[self._body_idxs], ref_traj.body_positions.shape
        )
        diff = ref_traj.body_positions - xpos_broadcast  # [:, self._body_idxs]

        return diff.flatten()

    def get_reference_rel_root_pos_local(self, data, ref_traj):
        """Reference position relative to current root position in root frame."""
        diff = ref_traj.position - data.qpos[:3]
        obs = self.global_vector_to_local_frame(data, diff)
        return jp.concatenate([o.flatten() for o in obs])

    def get_reference_rel_joints(self, data, ref_traj):
        """Observation of the reference joints relative to walker."""
        diff = (ref_traj.joints - data.qpos[7:])[:, self._joint_idxs]

        # diff = (qpos_ref - data.qpos[7:])[:,self._joint_idxs]
        return diff.flatten()

    def get_reference_appendages_pos(self, ref_traj):
        """Reference appendage positions in reference frame, not relative."""
        return ref_traj.body_positions[:, self._app_idx].flatten()

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


# MAX_END_STEP = 10000
# class RodentMultiClipTracking(PipelineEnv):
#     def __init__(
#         self,
#         params,
#         ref_steps: Sequence[int],
#         healthy_z_range=(0.05, 0.5),
#         reset_noise_scale=1e-3,
#         clip_length: int = 250,
#         sub_clip_length: int = 10,
#         ref_traj_length: int = 5,
#         termination_threshold: float = 5,
#         body_error_multiplier: float = 1.0,
#         min_steps: int = 10,
#         **kwargs,
#     ):
#         # body_idxs => walker_bodies => body_positions
#         root = mjcf.from_path("./assets/rodent.xml")

#         # TODO: replace this rescale with jax version (from james cotton BodyModels)
#         rescale.rescale_subtree(
#             root,
#             params["scale_factor"],
#             params["scale_factor"],
#         )
#         mj_model = mjcf.Physics.from_mjcf_model(root).model.ptr

#         mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

#         mj_model.opt.solver = {
#             "cg": mujoco.mjtSolver.mjSOL_CG,
#             "newton": mujoco.mjtSolver.mjSOL_NEWTON,
#         }[params["solver"].lower()]
#         mj_model.opt.iterations = params["iterations"]
#         mj_model.opt.ls_iterations = params["ls_iterations"]
#         mj_model.opt.jacobian = 0  # dense

#         self._end_eff_idx = jp.array(
#             [
#                 mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
#                 for body in params["end_eff_names"]
#             ]
#         )

#         self._body_idxs = jp.array(
#             [
#                 mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
#                 for body in params["walker_body_names"]
#             ]
#         )

#         self._joint_idxs = jp.array(
#             [
#                 mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
#                 for joint in params["joint_names"]
#             ]
#         )

#         sys = mjcf_brax.load_model(mj_model)

#         physics_steps_per_control_step = 5

#         kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
#         kwargs["backend"] = "mjx"

#         super().__init__(sys, **kwargs)

#         self._healthy_z_range = healthy_z_range
#         self._reset_noise_scale = reset_noise_scale
#         self._termination_threshold = termination_threshold
#         self._body_error_multiplier = body_error_multiplier
#         self._clip_length = clip_length
#         self._sub_clip_length = sub_clip_length
#         self._ref_traj_length = ref_traj_length
#         self._termination_threshold = termination_threshold
#         self._body_error_multiplier = body_error_multiplier

#         self._ref_steps = np.sort(ref_steps)
#         self._max_ref_step = self._ref_steps[-1]
#         self._min_steps = min_steps

#         with open(params["clip_path"], "rb") as f:
#             self._ref_traj = pickle.load(f)

#         if self._sub_clip_length > self._clip_length:
#             raise ValueError("sub clip length cannot be greater than clip_length!")

#     def _load_reference_data(
#         self, ref_path, proto_modifier, dataset: mp.ClipCollection
#     ):
#         """load dataset from the data class ClipCollections in mp"""

#         # This is how it was called in validation
#         from dm_control.locomotion.tasks.reference_pose import types
#         from dm_control.utils import io as resources

#         file_name = "clips/..."
#         current_directory = os.getcwd()
#         TEST_FILE_PATH = os.path.join(current_directory, file_name)

#         with h5py.File(TEST_FILE_PATH, "r") as f:
#             dataset_keys = tuple(f.keys())
#             dataset = types.ClipCollection(
#                 ids=dataset_keys,
#             )

#         ref_path = resources.GetResourceFilename(TEST_FILE_PATH)

#         # TODO: what is the relevant for loading .p directly? and what is proto_modifier? loader is a big class in dm_control
#         self._loader = loader.HDF5TrajectoryLoader(
#             ref_path, proto_modifier=proto_modifier
#         )  # loader is used to get rajectory from id provided by ClipCollections class

#         self._dataset = dataset
#         self._num_clips = len(self._dataset.ids)

#         if self._dataset.end_steps is None:
#             # load all trajectories to infer clip end steps.
#             self._all_clips = [
#                 self._loader.get_trajectory(
#                     clip_id, start_step=clip_start_step, end_step=_MAX_END_STEP
#                 )
#                 for clip_id, clip_start_step in zip(
#                     self._dataset.ids, self._dataset.start_steps
#                 )
#             ]
#             # infer clip end steps to set sampling distribution
#             self._dataset.end_steps = tuple(clip.end_step for clip in self._all_clips)
#         else:
#             self._all_clips = [None] * self._num_clips

#     def _get_possible_starts(self):
#         """
#         self._possible_starts is all possible (clip, step) starting points
#         """

#         self._possible_starts = []
#         self._start_probabilities = []
#         dataset = self._dataset

#         for clip_number, (start, end, weight) in enumerate(
#             zip(dataset.start_steps, dataset.end_steps, dataset.weights)
#         ):
#             # length - required lookahead - minimum number of steps
#             last_possible_start = end - self._max_ref_step - self._min_steps

#             if self._always_init_at_clip_start:
#                 self._possible_starts += [(clip_number, start)]
#                 self._start_probabilities += [weight]
#             else:
#                 self._possible_starts += [
#                     (clip_number, j) for j in range(start, last_possible_start)
#                 ]
#                 self._start_probabilities += [
#                     weight for _ in range(start, last_possible_start)
#                 ]

#         # normalize start probabilities
#         self._start_probabilities = np.array(self._start_probabilities) / np.sum(
#             self._start_probabilities
#         )

#     def _get_clip_to_track(self, random_state: np.random.RandomState):
#         """
#         main muticlip selection function
#         1. self._possible_starts stores all (clip_index, start_step)
#         2. self._start_probabilities keeps weighted clip's prob
#         """
#         # get specific clip index and start frame
#         index = random_state.choice(
#             len(self._possible_starts), p=self._start_probabilities
#         )
#         clip_index, start_step = self._possible_starts[index]
#         self._current_clip_index = clip_index

#         # get clip id
#         clip_id = self._dataset.ids[self._current_clip_index]

#         # TODO: fetch selected trajectory from loader?
#         if self._all_clips[self._current_clip_index] is None:
#             self._all_clips[self._current_clip_index] = self._loader.get_trajectory(
#                 clip_id,
#                 start_step=self._dataset.start_steps[self._current_clip_index],
#                 end_step=self._dataset.end_steps[self._current_clip_index],
#                 zero_out_velocities=False,
#             )
#             self._current_clip = self._all_clips[
#                 self._current_clip_index
#             ]  # this is where you get the current ref_traj

#         self._time_step = (
#             start_step - self._dataset.start_steps[self._current_clip_index]
#         )

#         self._current_start_time = (
#             start_step - self._dataset.start_steps[self._current_clip_index]
#         ) * self._current_clip.dt

#         # TODO: not sure what this does
#         self._last_step = (
#             len(self._clip_reference_features["joints"]) - self._max_ref_step - 1
#         )

#     def reset(self, rng) -> State:
#         """
#         Resets the environment to an initial state.
#         TODO: add a small amt of noise (qpos + epsilon) for randomization purposes
#         """
#         start_frame = jax.random.randint(
#             rng,
#             (),
#             0,
#             self._clip_length - self._sub_clip_length - self._ref_traj_length,
#         )
#         old, rng = jax.random.split(rng)

#         # TODO: use self._current_start_time for start_frame? use self._current_clip for self._ref_traj?
#         self._get_clip_to_track(rng)

#         noise = self._reset_noise_scale * jax.random.normal(rng, shape=(self.sys.nq,))

#         qpos = jp.hstack(
#             [
#                 self._ref_traj.position[start_frame, :],
#                 self._ref_traj.quaternion[start_frame, :],
#                 self._ref_traj.joints[start_frame, :],
#             ]
#         )
#         qvel = jp.hstack(
#             [
#                 self._ref_traj.velocity[start_frame, :],
#                 self._ref_traj.angular_velocity[start_frame, :],
#                 self._ref_traj.joints_velocity[start_frame, :],
#             ]
#         )
#         data = self.pipeline_init(qpos + noise, qvel)
#         traj = self._get_traj(data, start_frame)

#         info = {
#             "cur_frame": start_frame,
#             "traj": traj,
#             "first_reset": 0,
#         }
#         obs = self._get_obs(data, jp.zeros(self.sys.nu), info)
#         reward, done, zero = jp.zeros(3)
#         metrics = {
#             "rcom": zero,
#             "rvel": zero,
#             "rtrunk": zero,
#             "rquat": zero,
#             "ract": zero,
#             "rapp": zero,
#             "termination_error": zero,
#         }

#         state = State(data, obs, reward, done, metrics, info)
#         termination_error = self._calculate_termination(state)
#         info["termination_error"] = termination_error
#         # if termination_error > 1e-1:
#         #   raise ValueError(('The termination exceeds 1e-2 at initialization. '
#         #                     'This is likely due to a proto/walker mismatch.'))
#         state = state.replace(info=info)

#         return state