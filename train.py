import functools
import jax
from jax import numpy as jp
from typing import Dict
import wandb
import numpy as np
from brax import envs
from brax.io import model

import hydra
from omegaconf import DictConfig, OmegaConf
import mujoco
import imageio

# from brax.training.agents.ppo import train as brax_ppo
from brax.training.agents.ppo import networks as brax_networks
from ppo_imitation import train as ppo
from ppo_imitation import ppo_networks as custom_ppo_networks

from envs.rodent import RodentTracking
from envs.cmu_humanoid import CMUHumanoidRun
from typing import Union
from brax import envs
from brax.v1 import envs as envs_v1

import numpy as np
import uuid
import pickle
from preprocessing.mjx_preprocess import process_clip_to_train

State = Union[envs.State, envs_v1.State]
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["NUMEXPR_MAX_THREADS"] = "64"


def jax_has_gpu():
    try:
        _ = jax.device_put(jp.ones(1), device=jax.devices("gpu")[0])
        return True
    except:
        return False


if jax_has_gpu():
    n_devices = jax.device_count(backend="gpu")
    print(f"Using {n_devices} GPUs")
else:
    n_devices = 1
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)

envs.register_environment("rodent", RodentTracking)
envs.register_environment("cmuhumanoidrun", CMUHumanoidRun)


@hydra.main(config_path="./configs", config_name="train_config", version_base=None)
def main(train_config: DictConfig):
    env_cfg = hydra.compose(config_name="env_config")
    env_cfg = OmegaConf.to_container(env_cfg, resolve=True)
    env_cfg = env_cfg[train_config.env_name]
    env_args = env_cfg["env_args"]

    if train_config.env_name == "rodent":
        reference_path = f"clips/{env_cfg['clip_idx']}.p"

        if os.path.exists(reference_path):
            with open(reference_path, "rb") as file:
                # Use pickle.load() to load the data from the file
                reference_clip = pickle.load(file)
        else:
            # Process rodent clip and save as pickle
            reference_clip = process_clip_to_train(
                env_cfg["stac_path"],
                start_step=env_cfg["clip_idx"] * env_args["clip_length"],
                clip_length=env_args["clip_length"],
                mjcf_path=env_args["mjcf_path"],
            )
            with open(reference_path, "wb") as file:
                # Use pickle.dump() to save the data to the file
                pickle.dump(reference_clip, file)
    else:
        reference_clip = None

    # Init env
    env = envs.get_environment(
        env_cfg["name"],
        reference_clip=reference_clip,
        **env_args,
    )

    # Set the env to always start at frame 0 by maximizing sub_clip_length
    eval_env_args = env_args.copy()
    eval_env_args["sub_clip_length"] = (
        env_args["clip_length"] - env_args["ref_traj_length"]
    )
    eval_env = envs.get_environment(
        env_cfg["name"],
        reference_clip=reference_clip,
        **eval_env_args,
    )

    jit_step = jax.jit(eval_env.step)
    jit_reset = jax.jit(eval_env.reset)

    if train_config["algo_name"] == "custom_ppo":
        train = ppo.custom_train
        if train_config["policy_network_name"] == "mlp":
            network_factory = functools.partial(
                custom_ppo_networks.make_mlp_ppo_networks,
                policy_layer_sizes=train_config.mlp_policy_layer_sizes,
            )
        else:
            raise ValueError("invalid config: policy_network_name")
    elif train_config["algo_name"] == "brax_ppo":
        train = ppo.brax_train
        network_factory = functools.partial(
            brax_networks.make_ppo_networks,
            policy_hidden_layer_sizes=train_config.mlp_policy_layer_sizes,
            value_hidden_layer_sizes=(256, 256),
        )
    else:
        raise ValueError(f"unsupported algo name: {train_config['algo_name']}")

    train_fn = functools.partial(
        train,
        num_timesteps=train_config["num_timesteps"],
        num_evals=int(train_config["num_timesteps"] / train_config["eval_every"]),
        reward_scaling=1,
        episode_length=train_config["episode_length"],
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=train_config["num_minibatches"],
        num_updates_per_batch=train_config["num_updates_per_batch"],
        discounting=0.95,
        learning_rate=train_config["learning_rate"],
        entropy_cost=train_config["entropy_cost"],
        num_envs=train_config["num_envs"] * n_devices,
        batch_size=train_config["batch_size"] * n_devices,
        seed=0,
        clipping_epsilon=train_config["clipping_epsilon"],
        network_factory=network_factory,
        deterministic_eval=True,
    )

    # Generates a completely random UUID (version 4)
    run_id = uuid.uuid4()
    model_path = f"./model_checkpoints/{run_id}"

    merged_conf = OmegaConf.merge(env_cfg, train_config)

    run = wandb.init(
        project="VNL_SingleClipImitationPPO_Intention",
        config=OmegaConf.to_container(merged_conf, resolve=True),
        notes=train_config["note"],
        dir="/tmp",
    )

    wandb.run.name = f"{train_config.env_name}_{train_config.task_name}_{train_config['algo_name']}_{run_id}"

    def wandb_progress(num_steps, metrics):
        metrics["num_steps"] = num_steps
        wandb.log(metrics, commit=False)

    # TODO: make the rollout into a scan (or call brax's rollout fn?)
    # wandb.log commit=False until the last log (probably the video)
    def policy_params_fn(
        num_steps,
        make_policy,
        policy_params,
        value_apply,
        value_params,
        model_path=model_path,
    ):
        os.makedirs(model_path, exist_ok=True)
        model.save_params(f"{model_path}/{num_steps}", policy_params)
        jit_inference_fn = jax.jit(make_policy(policy_params, deterministic=True))
        jit_value_apply = jax.jit(value_apply)
        reset_rng, act_rng = jax.random.split(jax.random.PRNGKey(0))

        state = jit_reset(reset_rng)

        rollout = [state.pipeline_state]
        errors = []
        rewards = []
        means = []
        actions = []
        log_probs = []
        ctrls = []
        z_heights = []
        values = []
        for i in range(eval_env._clip_length):
            _, act_rng = jax.random.split(act_rng)
            if train_config["algo_name"] == "custom_ppo":
                obs = jp.concatenate([state.obs, state.info["traj"]], axis=-1)
                ctrl, extras = jit_inference_fn(obs, act_rng)
                value = jit_value_apply(policy_params[0], value_params, obs)
            elif train_config["algo_name"] == "brax_ppo":
                obs = state.obs
                ctrl, extras = jit_inference_fn(obs, act_rng)
                value = jit_value_apply(policy_params[0], value_params, obs)
            else:
                raise ValueError(f"unsupported algo name: {train_config['algo_name']}")
            
            state = jit_step(state, ctrl)

            # mean = extras["logits"]
            # log_prob = extras["log_prob"]
            # action = extras["actions"]
            # log_probs.append(log_prob)
            # actions.append(action)
            # means.append(mean)
            values.append(value)
            rewards.append(state.reward)
            ctrls.append(ctrl)
            rollout.append(state.pipeline_state)
            z_heights.append(state.pipeline_state.xpos[eval_env._torso_idx][2])

        # Plot normalizer params
        data = [[c] for c in policy_params[0].mean.flatten()]
        table = wandb.Table(data=data, columns=["running_statistics means"])
        wandb.log(
            {
                f"logits/running_statistics_means": wandb.plot.histogram(
                    table, "running_statistics means", title="obs normalizer means"
                )
            },
            commit=False,
        )

        # Plot normalizer params
        data = [[c] for c in policy_params[0].std.flatten()]
        table = wandb.Table(data=data, columns=["running_statistics stds"])
        wandb.log(
            {
                f"logits/running_statistics_stds": wandb.plot.histogram(
                    table, "running_statistics stds", title="obs normalizer stds"
                )
            },
            commit=False,
        )

        # Plot value estimate over rollout
        data = [[x, y] for (x, y) in zip(range(len(values)), values)]
        table = wandb.Table(data=data, columns=["frame", "values"])
        wandb.log(
            {
                "eval/rollout_value_estimate": wandb.plot.line(
                    table,
                    "frame",
                    "values",
                    title="value estimate for each rollout frame",
                )
            },
            commit=False,
        )

        # Plot rtrunk over rollout
        data = [[x, y] for (x, y) in zip(range(len(errors)), errors)]
        table = wandb.Table(data=data, columns=["frame", "rtrunk"])
        wandb.log(
            {
                "eval/rollout_rtrunk": wandb.plot.line(
                    table,
                    "frame",
                    "rtrunk",
                    title="rtrunk for each rollout frame",
                )
            },
            commit=False,
        )

        # Plot z height over rollout
        data = [[x, y] for (x, y) in zip(range(len(z_heights)), z_heights)]
        table = wandb.Table(data=data, columns=["frame", "z height"])
        wandb.log(
            {
                "eval/rollout_rtrunk": wandb.plot.line(
                    table,
                    "frame",
                    "z height",
                    title="z height for each rollout frame",
                )
            },
            commit=False,
        )

        # Plot action means over rollout (array of array)
        data = [[c] for c in list(np.array(ctrls).flatten())]
        table = wandb.Table(data=data, columns=["actions"])
        wandb.log(
            {
                f"logits/rollout_actions": wandb.plot.histogram(
                    table, "actions", title="Final Action Distribution"
                )
            },
            commit=False,
        )

        # # Plot action means over rollout (array of array)
        # data = np.array(actions).T
        # wandb.log(
        #     {
        #         f"logits/rollout_actions": wandb.plot.line_series(
        #             xs=range(data.shape[1]),
        #             ys=data,
        #             keys=[str(i) for i in range(data.shape[0])],
        #             xname="Frame",
        #             title=f"Action actuator means for each rollout frame (post-processed)",
        #         )
        #     }
        # )

        # # Plot policy action prob over rollout
        # data = [[x, y] for (x, y) in zip(range(len(log_probs)), log_probs)]
        # table = wandb.Table(data=data, columns=["frame", "log_probs"])
        # wandb.log(
        #     {
        #         "logits/rollout_log_probs": wandb.plot.line(
        #             table,
        #             "frame",
        #             "log_probs",
        #             title="Policy action probability for each rollout frame",
        #         )
        #     }
        # )

        # Plot reward over rollout
        data = [[x, y] for (x, y) in zip(range(len(rewards)), rewards)]
        table = wandb.Table(data=data, columns=["frame", "reward"])
        wandb.log(
            {
                "eval/rollout_reward": wandb.plot.line(
                    table,
                    "frame",
                    "reward",
                    title="reward for each rollout frame",
                )
            },
            commit=False,
        )

        # Render the walker with the reference expert demonstration trajectory
        os.environ["MUJOCO_GL"] = "osmesa"

        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    0,
                    eval_env._clip_length,
                )
            return jp.array([])

        qposes_rollout = [data.qpos for data in rollout]
        
        if train_config.env_name == "rodent":
            # extract qpos from rollout
            ref_traj = eval_env._ref_traj
            ref_traj = jax.tree_util.tree_map(f, ref_traj)
            qposes_ref = jp.hstack(
                [ref_traj.position, ref_traj.quaternion, ref_traj.joints]
            )
        else:
            qposes_ref = qposes_rollout

        mj_model = mujoco.MjModel.from_xml_path(f"./assets/{env_cfg['rendering_mjcf']}")

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }["cg"]

        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.jacobian = 0  # dense
        mj_data = mujoco.MjData(mj_model)

        # save rendering and log to wandb
        os.environ["MUJOCO_GL"] = "osmesa"
        mujoco.mj_kinematics(mj_model, mj_data)
        renderer = mujoco.Renderer(mj_model, height=512, width=512)

        frames = []
        # render while stepping using mujoco
        video_path = f"{model_path}/{num_steps}.mp4"

        with imageio.get_writer(video_path, fps=float(1.0 / eval_env.dt)) as video:
            for qpos1, qpos2 in zip(qposes_ref, qposes_rollout):
                if train_config.env_name == "rodent":
                    mj_data.qpos = np.append(qpos1, qpos2)
                else:
                    mj_data.qpos = np.append(qpos2, qpos2)

                mujoco.mj_forward(mj_model, mj_data)

                renderer.update_scene(mj_data, camera=f"{env_cfg['camera']}")

                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_progress,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env,
    )

    final_save_path = f"{model_path}/finished"
    model.save_params(final_save_path, params)
    print(f"Run finished. Model saved to {final_save_path}")


if __name__ == "__main__":
    main()
