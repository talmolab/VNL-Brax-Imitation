MAX_END_STEP = 10000
class RodentMultiClipTracking(PipelineEnv):
    def __init__(
        self,
        params,
        ref_steps: Sequence[int],
        healthy_z_range=(0.05, 0.5),
        reset_noise_scale=1e-3,
        clip_length: int = 250,
        sub_clip_length: int = 10,
        ref_traj_length: int = 5,
        termination_threshold: float = 5,
        body_error_multiplier: float = 1.0,
        min_steps: int = 10,
        **kwargs,
    ):
        # body_idxs => walker_bodies => body_positions
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
        self._reset_noise_scale = reset_noise_scale
        self._termination_threshold = termination_threshold
        self._body_error_multiplier = body_error_multiplier
        self._clip_length = clip_length
        self._sub_clip_length = sub_clip_length
        self._ref_traj_length = ref_traj_length
        self._termination_threshold = termination_threshold
        self._body_error_multiplier = body_error_multiplier

        self._ref_steps = np.sort(ref_steps)
        self._max_ref_step = self._ref_steps[-1]
        self._min_steps = min_steps

        with open(params["clip_path"], "rb") as f:
            self._ref_traj = pickle.load(f)

        if self._sub_clip_length > self._clip_length:
            raise ValueError("sub clip length cannot be greater than clip_length!")

    def _load_reference_data(
        self, ref_path, proto_modifier, dataset: mp.ClipCollection
    ):
        """load dataset from the data class ClipCollections in mp"""

        # This is how it was called in validation
        from dm_control.locomotion.tasks.reference_pose import types
        from dm_control.utils import io as resources

        file_name = "clips/..."
        current_directory = os.getcwd()
        TEST_FILE_PATH = os.path.join(current_directory, file_name)

        with h5py.File(TEST_FILE_PATH, "r") as f:
            dataset_keys = tuple(f.keys())
            dataset = types.ClipCollection(
                ids=dataset_keys,
            )

        ref_path = resources.GetResourceFilename(TEST_FILE_PATH)

        # TODO: what is the relevant for loading .p directly? and what is proto_modifier? loader is a big class in dm_control
        self._loader = loader.HDF5TrajectoryLoader(
            ref_path, proto_modifier=proto_modifier
        )  # loader is used to get rajectory from id provided by ClipCollections class

        self._dataset = dataset
        self._num_clips = len(self._dataset.ids)

        if self._dataset.end_steps is None:
            # load all trajectories to infer clip end steps.
            self._all_clips = [
                self._loader.get_trajectory(
                    clip_id, start_step=clip_start_step, end_step=_MAX_END_STEP
                )
                for clip_id, clip_start_step in zip(
                    self._dataset.ids, self._dataset.start_steps
                )
            ]
            # infer clip end steps to set sampling distribution
            self._dataset.end_steps = tuple(clip.end_step for clip in self._all_clips)
        else:
            self._all_clips = [None] * self._num_clips

    def _get_possible_starts(self):
        """
        self._possible_starts is all possible (clip, step) starting points
        """

        self._possible_starts = []
        self._start_probabilities = []
        dataset = self._dataset

        for clip_number, (start, end, weight) in enumerate(
            zip(dataset.start_steps, dataset.end_steps, dataset.weights)
        ):
            # length - required lookahead - minimum number of steps
            last_possible_start = end - self._max_ref_step - self._min_steps

            if self._always_init_at_clip_start:
                self._possible_starts += [(clip_number, start)]
                self._start_probabilities += [weight]
            else:
                self._possible_starts += [
                    (clip_number, j) for j in range(start, last_possible_start)
                ]
                self._start_probabilities += [
                    weight for _ in range(start, last_possible_start)
                ]

        # normalize start probabilities
        self._start_probabilities = np.array(self._start_probabilities) / np.sum(
            self._start_probabilities
        )

    def _get_clip_to_track(self, random_state: np.random.RandomState):
        """
        main muticlip selection function
        1. self._possible_starts stores all (clip_index, start_step)
        2. self._start_probabilities keeps weighted clip's prob
        """
        # get specific clip index and start frame
        index = random_state.choice(
            len(self._possible_starts), p=self._start_probabilities
        )
        clip_index, start_step = self._possible_starts[index]
        self._current_clip_index = clip_index

        # get clip id
        clip_id = self._dataset.ids[self._current_clip_index]

        # TODO: fetch selected trajectory from loader?
        if self._all_clips[self._current_clip_index] is None:
            self._all_clips[self._current_clip_index] = self._loader.get_trajectory(
                clip_id,
                start_step=self._dataset.start_steps[self._current_clip_index],
                end_step=self._dataset.end_steps[self._current_clip_index],
                zero_out_velocities=False,
            )
            self._current_clip = self._all_clips[
                self._current_clip_index
            ]  # this is where you get the current ref_traj

        self._time_step = (
            start_step - self._dataset.start_steps[self._current_clip_index]
        )

        self._current_start_time = (
            start_step - self._dataset.start_steps[self._current_clip_index]
        ) * self._current_clip.dt

        # TODO: not sure what this does
        self._last_step = (
            len(self._clip_reference_features["joints"]) - self._max_ref_step - 1
        )

    def reset(self, rng) -> State:
        """
        Resets the environment to an initial state.
        TODO: add a small amt of noise (qpos + epsilon) for randomization purposes
        """
        start_frame = jax.random.randint(
            rng,
            (),
            0,
            self._clip_length - self._sub_clip_length - self._ref_traj_length,
        )
        old, rng = jax.random.split(rng)

        # TODO: use self._current_start_time for start_frame? use self._current_clip for self._ref_traj?
        self._get_clip_to_track(rng)

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