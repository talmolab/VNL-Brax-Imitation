import functools
import jax
from jax import numpy as jp
from typing import Dict
import wandb
import numpy as np
from brax import envs
# from brax.training.agents.ppo import train as ppo
from brax.io import model

import brax_ppo as ppo
import mujoco
import imageio

from envs.humanoid import HumanoidTracking

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
)

#TODO: Use hydra for configs
config = {
    "env_name": "humanoid",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 4096*n_gpus,
    "num_timesteps": 500_000_000,
    "eval_every": 25_000_000,
    "episode_length": 150, # This is the clip length
    "batch_size": 256*n_gpus,
    "learning_rate": 1e-4,
    "terminate_when_unhealthy": True,
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 6,
    "tau":0.3,
}

env_params = {
    "solver": config['solver'],
    "iterations": config['iterations'],
    "ls_iterations": config['ls_iterations'],
    "clip_path": "clips/humanoid_traj.p",
}

envs.register_environment(config["env_name"], HumanoidTracking)
env = envs.get_environment(config["env_name"], params=env_params)

train_fn = functools.partial(
    ppo.train, num_timesteps=config["num_timesteps"], num_evals=int(config["num_timesteps"]/config["eval_every"]),
    reward_scaling=1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1,
    unroll_length=20, num_minibatches=64, num_updates_per_batch=8,
    discounting=0.95, learning_rate=config["learning_rate"], entropy_cost=1e-3, num_envs=config["num_envs"],
    batch_size=config["batch_size"], seed=0
)

import uuid
# Generates a completely random UUID (version 4)
run_id = uuid.uuid4()
model_path = f"./model_checkpoints/{run_id}"

run = wandb.init(
    project="VNL_Huammnoid_Imitation_Debug",
    config=config,
    notes=f"{config['batch_size']} batchsize, " + 
        f"{config['solver']}, {config['iterations']}/{config['ls_iterations']}"
)

wandb.run.name = f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}"

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
    env = envs.get_environment(config["env_name"], params={
                            "solver": "cg",
                            "iterations": 6,
                            "ls_iterations": 6,
                            "clip_path": "humanoid_traj.p",
                            }
    )
    jit_step = jax.jit(env.step)
    state = env.reset_to_frame(0)
    rollout = [state.pipeline_state]
    act_rng = jax.random.PRNGKey(0)
    errors = []
    for _ in range(env._clip_length - env._ref_traj_length):
        _, act_rng = jax.random.split(act_rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        errors.append(state.info['termination_error'])
        rollout.append(state.pipeline_state)
        
    data = [[x, y] for (x, y) in zip(range(len(errors)), errors)]
    table = wandb.Table(data=data, columns=["frame", "frame termination error"])
    wandb.log(
        {
            "eval/rollout_termination_error": wandb.plot.line(
                table, "frame",  "frame termination error", title="Termination error for each rollout frame"
            )
        }
    )    
    
    # save rendering and log to wandb
    os.environ["MUJOCO_GL"] = "osmesa"
    
    video_path = f"{model_path}/{num_steps}.mp4"
    with imageio.get_writer(video_path, fps=30.0) as video:
        imgs = env.render(rollout, camera="side", height=512, width=512)
        for i, im in enumerate(imgs):
            video.append_data(im)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
    
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn)

final_save_path = f"{model_path}/finished_mlp"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")
