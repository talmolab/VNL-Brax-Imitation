import functools
import jax
from typing import Dict
import wandb

from brax import envs
from ppo_imitation.train import train
from brax.io import model

from environments import RodentSingleClipTrack

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

#TODO: Use hydra for configs
config = {
    "env_name": "rodent_single_clip",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 2048*n_gpus,
    "num_timesteps": 100_000_000,
    "eval_every": 5_000_000,
    "episode_length": 1000,
    "batch_size": 2048*n_gpus,
    "learning_rate": 5e-5,
    "terminate_when_unhealthy": True,
    "run_platform": "Salk",
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 3,
}

# # Preprocess step
# import mocap_preprocess as mp
# data_path = "/n/home05/charleszhang/stac-mjx/transform_snips.p"
# clip_paths = mp.process(data_path, 
#          "transform_snips_250.h5", 
#          n_steps=250,
#          ref_steps=(1,2,3,4,5))

params = {
    "scale_factor": .9,
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 3,
    "clip_path": "12_21_1_250_clip_0.p",
    "end_eff_names": [
        "foot_L",
        "foot_R",
        "hand_L",
        "hand_R",
    ],
}

envs.register_environment(config["env_name"], RodentSingleClipTrack)
env = envs.get_environment(config["env_name"], params = params)

train_fn = functools.partial(
    train, num_timesteps=config["num_timesteps"], num_evals=int(config["num_timesteps"]/config["eval_every"]),
    reward_scaling=1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=64, num_updates_per_batch=8,
    discounting=0.99, learning_rate=config["learning_rate"], entropy_cost=1e-3, num_envs=config["num_envs"],
    batch_size=config["batch_size"], seed=0
)

import uuid
# Generates a completely random UUID (version 4)
run_id = uuid.uuid4()
model_path = f"./model_checkpoints/{run_id}"

run = wandb.init(
    project="VNL_SingleClipImitationPPO",
    config=config,
    notes=f"{config['batch_size']} batchsize, " + 
        f"{config['solver']}, {config['iterations']}/{config['ls_iterations']}"
)

wandb.run.name = f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}"

def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics)
    
def policy_params_fn(num_steps, make_policy, params, model_path=model_path):
    os.makedirs(model_path, exist_ok=True)
    model.save_params(f"{model_path}/{num_steps}", params)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn)

final_save_path = f"{model_path}/finished_mlp"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")