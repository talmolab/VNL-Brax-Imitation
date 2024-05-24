import functools
import jax
from typing import Dict
import wandb

from brax import envs
# from ppo_imitation.train import train
from brax.training.agents.ppo import train as ppo
from brax.io import model

from environments import RodentSingleClipTrack
from environments_debug import HumanoidTracking

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os

import mujoco
import imageio

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
    "env_name": "humannoid_debug", #"rodent_single_clip",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 2048*n_gpus,
    "num_timesteps": 100_000_000,
    "eval_every": 5_000_000,
    "episode_length": 150,
    "batch_size": 2048*n_gpus,
    "learning_rate": 5e-5,
    "terminate_when_unhealthy": True,
    "run_platform": "Salk",
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 3,
    'num_eval_envs': 1, # TODO: averages over eval_env?
    'num_resets_per_eval': 0, # carry over state object
    'unroll_length': 10
}

# # Preprocess step
# import mocap_preprocess as mp
# data_path = "all_snips.p"
# clip_paths = mp.process(data_path, 
#          "all_snips_250_clip_500.p", 
#          n_steps=750,
#          ref_steps=(1,2,3,4,5))

# env_params = {
#     "scale_factor": .9,
#     "solver": "cg",
#     "iterations": 6,
#     "ls_iterations": 3,
#     "clip_path": "all_snips_250_clip_500.p",
#     "end_eff_names": [
#         "foot_L",
#         "foot_R",
#         "hand_L",
#         "hand_R",
#     ],
# }

env_params = {
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 6,
    "clip_path": "humanoid_traj.p",
}

# envs.register_environment(config["env_name"], RodentSingleClipTrack)
envs.register_environment(config["env_name"], HumanoidTracking)
env = envs.get_environment(config["env_name"], params = env_params)

train_fn = functools.partial(
    ppo.train, num_timesteps=config["num_timesteps"],
    num_evals=int(config["num_timesteps"]/config["eval_every"]),
    num_eval_envs=config['num_eval_envs'],
    num_resets_per_eval=config['num_resets_per_eval'],
    unroll_length=config['unroll_length'],
    reward_scaling=1,
    episode_length=config["episode_length"],
    normalize_observations=True,
    action_repeat=1,
    num_minibatches=64,
    num_updates_per_batch=8,
    discounting=0.99,
    learning_rate=config["learning_rate"],
    entropy_cost=1e-3,
    num_envs=config["num_envs"],
    batch_size=config["batch_size"],
    seed=123
)

import uuid
# Generates a completely random UUID (version 4)
run_id = uuid.uuid4()
model_path = f"./model_checkpoints/{run_id}"

run = wandb.init(
    project="VNL_Huammnoid_Imitation_Debug", #"VNL_SingleClipImitationPPO",
    config=config,
    notes=f"{config['batch_size']} batchsize, " + 
        f"{config['solver']}, {config['iterations']}/{config['ls_iterations']}"
)

wandb.run.name = f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}"

def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics)
    
# def policy_params_fn(num_steps, make_policy, params, model_path=model_path):
#     os.makedirs(model_path, exist_ok=True)
#     model.save_params(f"{model_path}/{num_steps}", params)

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
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        # print(ctrl)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

# save rendering and log to wandb
    os.environ["MUJOCO_GL"] = "osmesa"

    video_path = f"{model_path}/{num_steps}.mp4"
    with imageio.get_writer(video_path, fps=50.0) as video:
        imgs = env.render(rollout, camera='close_profile', height=512, width=512)
        for im in imgs:
            video.append_data(im)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn)

final_save_path = f"{model_path}/finished_mlp"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")