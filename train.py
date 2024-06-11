import functools
import jax
from jax import numpy as jp
from typing import Dict
import wandb
import numpy as np
from brax import envs
import brax_ppo as ppo
from brax.io import model

import hydra
from omegaconf import DictConfig, OmegaConf

import mujoco
import imageio

from envs.humanoid import HumanoidTracking, HumanoidStanding
from envs.ant import AntTracking
from envs.rodent import RodentTracking

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

envs.register_environment("humanoidtracking", HumanoidTracking)
envs.register_environment("ant", AntTracking)
envs.register_environment("rodent", RodentTracking)
envs.register_environment("humanoidstanding", HumanoidStanding)
    
@hydra.main(config_path="./configs", config_name="train_config", version_base=None)
def main(train_config: DictConfig):
    cfg = hydra.compose(config_name="env_config")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    env = envs.get_environment(cfg[train_config.env_name]["name"], params=cfg[train_config.env_name])

    train_fn = functools.partial(
        ppo.train, num_timesteps=train_config["num_timesteps"], num_evals=int(train_config["num_timesteps"]/train_config["eval_every"]),
        reward_scaling=1, episode_length=train_config["episode_length"], normalize_observations=True, action_repeat=1,
        unroll_length=20, num_minibatches=64, num_updates_per_batch=8,
        discounting=0.99, learning_rate=train_config["learning_rate"], entropy_cost=1e-3, num_envs=train_config["num_envs"]*n_gpus,
        batch_size=train_config["batch_size"]*n_gpus, seed=0, clipping_epsilon=train_config["clipping_epsilon"]
    )

    import uuid
    # Generates a completely random UUID (version 4)
    run_id = uuid.uuid4()
    model_path = f"./model_checkpoints/{run_id}"

    run = wandb.init(
        project="VNL_SingleClipImitationPPO",
        config=OmegaConf.to_container(train_config, resolve=True),
        notes=f""
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
        env = envs.get_environment(cfg[train_config.env_name]["name"], params=cfg[train_config.env_name])
        
        jit_step = jax.jit(env.step)
        state = env.reset_to_frame(0)
        rollout = [state.pipeline_state]
        act_rng = jax.random.PRNGKey(0)
        errors = []
        for _ in range(train_config["episode_length"]):
            _, act_rng = jax.random.split(act_rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            if train_config.env_name != "humanoidstanding":
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
        with imageio.get_writer(video_path, fps=float(1.0 / env.dt)) as video:
            imgs = env.render(rollout, camera=cfg[train_config.env_name]['camera'], height=512, width=512)
            for i, im in enumerate(imgs):
                # print(im)
                video.append_data(im)

        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
        
    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn)

    final_save_path = f"{model_path}/finished_mlp"
    model.save_params(final_save_path, params)
    print(f"Run finished. Model saved to {final_save_path}")

if __name__ == "__main__":
    main()
