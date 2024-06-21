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

from ppo_imitation import train as ppo
from ppo_imitation import ppo_networks

from envs.humanoid import HumanoidTracking, HumanoidStanding
from envs.ant import AntTracking
from envs.rodent import RodentTracking

from typing import Sequence, Tuple, Union
import brax
from brax import envs
from brax.training.types import Policy
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.v1 import envs as envs_v1
import numpy as np
import uuid

State = Union[envs.State, envs_v1.State]
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


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

envs.register_environment("humanoidtracking", HumanoidTracking)
envs.register_environment("ant", AntTracking)
envs.register_environment("rodent", RodentTracking)
envs.register_environment("humanoidstanding", HumanoidStanding)


@hydra.main(config_path="./configs", config_name="train_config", version_base=None)
def main(train_config: DictConfig):
    cfg = hydra.compose(config_name="env_config")
    cfg = OmegaConf.to_container(cfg, resolve=True)

    env = envs.get_environment(
        cfg[train_config.env_name]["name"], params=cfg[train_config.env_name]
    )

    # TODO: make the intention network factory a part of the config
    intention_network_factory = functools.partial(
        ppo_networks.make_intention_ppo_networks,
        intention_latent_size=train_config.intention_latent_size,
        encoder_layer_sizes=train_config.encoder_layer_sizes,
        decoder_layer_sizes=train_config.decoder_layer_sizes,
    )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=train_config["num_timesteps"],
        num_evals=int(train_config["num_timesteps"] / train_config["eval_every"]),
        reward_scaling=1,
        episode_length=train_config["episode_length"],
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=train_config["num_minibatches"],
        num_updates_per_batch=train_config["num_updates_per_batch"],
        discounting=0.99,
        learning_rate=train_config["learning_rate"],
        entropy_cost=1e-3,
        num_envs=train_config["num_envs"] * n_devices,
        batch_size=train_config["batch_size"] * n_devices,
        seed=0,
        clipping_epsilon=train_config["clipping_epsilon"],
        kl_weight=train_config["kl_weight"],
        network_factory=intention_network_factory,
    )

    # Generates a completely random UUID (version 4)
    run_id = uuid.uuid4()
    model_path = f"./model_checkpoints/{run_id}"

    run = wandb.init(
        project="VNL_SingleClipImitationPPO",
        config=OmegaConf.to_container(train_config, resolve=True),
        notes=f"",
    )

    wandb.run.name = f"{train_config.env_name}_{train_config.task_name}_{train_config['algo_name']}_{run_id}"

    def wandb_progress(num_steps, metrics):
        metrics["num_steps"] = num_steps
        wandb.log(metrics)

    # TODO: make the rollout into a scan (or call brax's rollout fn?)
    def policy_params_fn(num_steps, make_policy, params, model_path=model_path):
        os.makedirs(model_path, exist_ok=True)
        model.save_params(f"{model_path}/{num_steps}", params)
        # rollout starting from frame 0
        jit_inference_fn = jax.jit(make_policy(params, deterministic=False))
        env = envs.get_environment(
            cfg[train_config.env_name]["name"], params=cfg[train_config.env_name]
        )

        jit_step = jax.jit(env.step)
        state = env.reset_to_frame(0)
        rollout = [state.pipeline_state]
        act_rng = jax.random.PRNGKey(0)
        errors = []
        means = []
        stds = []
        for _ in range(train_config["episode_length"]):
            _, act_rng = jax.random.split(act_rng)
            ctrl, extras = jit_inference_fn(state.info["traj"], state.obs, act_rng)
            state = jit_step(state, ctrl)
            if train_config.env_name != "humanoidstanding":
                errors.append(state.info["termination_error"])
            mean, std = np.split(extras["logits"], 2)
            means.append(mean)
            stds.append(std)
            rollout.append(state.pipeline_state)

        # Plot rtrunk over rollout
        data = [[x, y] for (x, y) in zip(range(len(errors)), errors)]
        table = wandb.Table(data=data, columns=["frame", "frame rtrunk"])
        wandb.log(
            {
                "eval/rollout_rtrunk": wandb.plot.line(
                    table,
                    "Frame",
                    "Frame rtrunk",
                    title="rtrunk for each rollout frame",
                )
            }
        )

        # Plot action means over rollout
        data = np.array(means).T
        wandb.log(
            {
                f"logits/rollout_means": wandb.plot.line_series(
                    xs=range(data.shape[1]),
                    ys=data,
                    keys=[str(i) for i in range(data.shape[0])],
                    xname="Frame",
                    title=f"Action actuator means for each rollout frame",
                )
            }
        )

        # Plot action stds over rollout (optimize this later)
        data = np.array(stds).T
        wandb.log(
            {
                f"logits/rollout_stds": wandb.plot.line_series(
                    xs=range(data.shape[1]),
                    ys=data,
                    keys=[str(i) for i in range(data.shape[0])],
                    xname="Frame",
                    title=f"Action actuator stds for each rollout frame",
                )
            }
        )

        # Render the walker with the reference expert demonstration trajectory
        os.environ["MUJOCO_GL"] = "osmesa"

        # extract qpos from rollout
        ref_traj = env._ref_traj
        ref_traj = jax.tree_util.tree_map(
            lambda x: jax.lax.slice_in_dim(x, 0, train_config["episode_length"]),
            ref_traj,
        )
        qposes_ref = jp.hstack(
            [ref_traj.position, ref_traj.quaternion, ref_traj.joints]
        )

        qposes_rollout = [data.qpos for data in rollout]

        mj_model = mujoco.MjModel.from_xml_path(
            f"./assets/{cfg[train_config.env_name]['rendering_mjcf']}"
        )
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
        with imageio.get_writer(video_path, fps=float(1.0 / env.dt)) as video:
            for i, (qpos1, qpos2) in enumerate(zip(qposes_ref, qposes_rollout)):
                # Set keypoints
                # physics, mj_model = ctrl.set_keypoint_sites(physics, keypoint_sites, kps)
                mj_data.qpos = np.append(qpos1, qpos2)
                mujoco.mj_forward(mj_model, mj_data)

                renderer.update_scene(
                    mj_data, camera=f"{cfg[train_config.env_name]['camera']}-0"
                )
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})

    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
    )

    final_save_path = f"{model_path}/finished_mlp"
    model.save_params(final_save_path, params)
    print(f"Run finished. Model saved to {final_save_path}")


if __name__ == "__main__":
    main()
