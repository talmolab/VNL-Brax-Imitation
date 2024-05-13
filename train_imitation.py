import functools
import brax.training
import brax.training.acting
import jax
from typing import Dict
import wandb

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model

from environments import RodentSingleClipTrack
from ppo_imitation.train import train

import warnings

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
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


brax.training.acting.actor_step = actor_step

# END TAPE

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

# TODO: Use hydra for configs
config = {
    "env_name": "rodent_single_clip",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 2048 * n_gpus,
    "num_timesteps": 100_000_000,
    "eval_every": 5_000_000,
    "episode_length": 1000,
    "batch_size": 2048 * n_gpus,
    "learning_rate": 5e-5,
    "terminate_when_unhealthy": True,
    "run_platform": "Salk",
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 3,
}

params = {
    "scale_factor": 0.9,
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 3,
    "clip_path": "data/12_22_1_250_clip_0.p",
    "end_eff_names": [
        "foot_L",
        "foot_R",
        "hand_L",
        "hand_R",
    ],
}

envs.register_environment(config["env_name"], RodentSingleClipTrack)
env = envs.get_environment(config["env_name"], params=params)


# Inject our custom network factory here.
train_fn = functools.partial(
    train,
    num_timesteps=config["num_timesteps"],
    num_evals=int(config["num_timesteps"] / config["eval_every"]),
    reward_scaling=1,
    episode_length=config["episode_length"],
    normalize_observations=True, # temporary change to false
    action_repeat=1,
    unroll_length=10,
    num_minibatches=64,
    num_updates_per_batch=8,
    discounting=0.99,
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
    notes=f"Explicitly separate the",
)

wandb.run.name = (
    f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}"
)


def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics)


def policy_params_fn(num_steps, make_policy, params, model_path=model_path):
    os.makedirs(model_path, exist_ok=True)
    model.save_params(f"{model_path}/{num_steps}", params)


make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
)

final_save_path = f"{model_path}/finished_mlp"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")
