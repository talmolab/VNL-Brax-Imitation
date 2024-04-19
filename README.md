# Brax-Imitation
rodent imitation learning using brax

setup:
- jax
- brax
- mujoco-mjx
- flax
- optax
  
## Current Progress (04/18/2024)
below is my current progrses in each of the files. Bear in mind that our implementation needs to be fully compatible with jax, since much of the logic will be jitted. This means using jax.numpy (jnp) instead of numpy, and making sure there are no side-effects in our functions so they can be jitted properly. Read through the quickstart pages in the jax documentation [here](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html). However, some initial setup stuff that dmcontrol/mjcf would help with could work!

- `losses.py`: I copied over the brax losses.py file for PPO, because our imitation learning needs a KL divergence term in the loss for regularizing the variational component. I added the function (`kl_divergence`) and added it as a term in the total loss (line 184, 186)

- `networks.py`: I added a basic VAE implementation for our policy network, and used a brax's MLP for the value network (as outlined in the Mimic paper). There are probably some details in the network architecture that are not implementation yet--The VAE current doesn't have a stochastic layer at the end for example.

- `obs_util.py`:  This file contains some functions required for transforming the trajectories to be reletive to the current state of the agent (see the Mimic methods section). The logic is taken from [the dmcontrol tracking task](
https://github.com/google-deepmind/dm_control/blob/7a6a5309e3ef79a720081d6d90958a2fb78fd3fe/dm_control/locomotion/tasks/reference_pose/tracking.py#L604). Needs some edits to have to work properly.

- `ppo.py`: This is just brax's ppo file. Will we need to change it?

- `preprocessing.py`/`preprocessing_utils.py`: According to the Mimic paper, the input trajectories are more than just the qpos that we get from stac. So these files get the rest of the data. Not finished. In the end we want it in the h5 file format. refer to the npmp_embeddings file I shared for how it was previously implemented

- `RodentImitationEnv.py` The brax environment. mostly the same as RodentRun for now, I just removed some of the running related logic and added more obs. Need to implement the whole reward calculation, and some good way of loading the trajectory data as part of the obs

- `train.py`: the high level wandb stuff and actually running ppo


