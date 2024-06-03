import functools
import jax
from jax import numpy as jp
from typing import Dict
import wandb
import numpy as np
from brax import envs

# from brax.training.agents.ppo import train as ppo
from brax.io import model

import ppo_imitation
import mujoco
import imageio

from envs.humanoid import HumanoidTracking

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

## duck tape

from typing import Sequence, Tuple, Union
import brax
from brax import envs
from brax.training.types import Policy
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.v1 import envs as envs_v1
import jax
import numpy as np

State = Union[envs.State, envs_v1.State]
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect data."""
    actions, policy_extras = policy(env_state.info["traj"], env_state.obs, key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={
            "policy_extras": policy_extras,
            "state_extras": state_extras,
            "traj": env_state.info["traj"],
        },
    )


brax.training.acting.actor_step = actor_step

# END TAPE

n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)

# TODO: Use hydra for configs
config = {
    "env_name": "humanoid",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 4096 * n_gpus,
    "num_timesteps": 500_000_000,
    "eval_every": 25_000_000,
    "episode_length": 150,  # This is the clip length
    "batch_size": 256 * n_gpus,
    "learning_rate": 1e-4,
    "terminate_when_unhealthy": True,
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 6,
}

env_params = {
    "solver": config["solver"],
    "iterations": config["iterations"],
    "ls_iterations": config["ls_iterations"],
    "clip_path": "data/humanoid_traj.p",
}

envs.register_environment(config["env_name"], HumanoidTracking)
env = envs.get_environment(config["env_name"], params=env_params)

train_fn = functools.partial(
    ppo_imitation.train,
    num_timesteps=config["num_timesteps"],
    num_evals=int(config["num_timesteps"] / config["eval_every"]),
    reward_scaling=1,
    episode_length=config["episode_length"],
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=64,
    num_updates_per_batch=8,
    discounting=0.95,
    learning_rate=config["learning_rate"],
    entropy_cost=1e-3,
    num_envs=config["num_envs"],
    batch_size=config["batch_size"],
    seed=0,
)

import uuid

# Generates a completely random UUID (version 4)
run_id = uuid.uuid4()
model_path = f"./model_checkpoints/{run_id}"

run = wandb.init(
    project="VNL_SingleClipImitationPPO",
    config=config,
    notes=f"{config['batch_size']} batchsize, "
    + f"{config['solver']}, {config['iterations']}/{config['ls_iterations']}",
)

wandb.run.name = (
    f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}_debug"
)


def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics)


# TODO: make the rollout into a scan (or call brax's rollout fn?)
def policy_params_fn(num_steps, make_policy, params, model_path=model_path):
    os.makedirs(model_path, exist_ok=True)
    model.save_params(f"{model_path}/{num_steps}", params)
    # print(params)
    # rollout starting from frame 0
    jit_inference_fn = jax.jit(make_policy(params, deterministic=False))
    env = envs.get_environment(config["env_name"], params=env_params)
    jit_step = jax.jit(env.step)
    state = env.reset_to_frame(0)
    rollout = [state.pipeline_state]
    act_rng = jax.random.PRNGKey(0)
    for _ in range(env._clip_length - env._ref_traj_length):
        _, act_rng = jax.random.split(act_rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
    # save rendering and log to wandb
    os.environ["MUJOCO_GL"] = "osmesa"

    video_path = f"{model_path}/{num_steps}.mp4"
    with imageio.get_writer(video_path, fps=30.0) as video:
        imgs = env.render(rollout, camera="side", height=512, width=512)
        for i, im in enumerate(imgs):
            video.append_data(im)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})


make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
)

final_save_path = f"{model_path}/finished_mlp"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")
