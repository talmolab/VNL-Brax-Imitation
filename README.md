# Brax-Imitation
Rodent imitation learning using Brax and MJX

setup:
- jax
- brax
- mujoco-mjx
- flax
- optax
- dmcontrol
- h5py
- moviepy
  
## Current Progress (05/8/2024)
#### Preprocessing
The preprocessing step is fully implemented in `mocap_preprocess.py`, with a usage example in `process_traj.ipynb`, with `mocap_preprocess.process()` being the main function you interact with. Here's a quick rundown: Suppose you have a stac'd trajectory called `transform_snips.p`. You pass this into `process()` along with the name of the .h5 save file as the two required arguments (there are many other optional arguments, view the docstring for more info). After calling `process()`, you will end up with two new files: an .h5 save file in the same format that was used in MiMic, as well as a set of .p save files containing the same data but in the format currently being used by our new Brax imitation learning environment: the `ReferenceTrajectory` Jax dataclass defined in `mocap_preprocess.py` (This format is subject to change). The number of save files depends on the number of clips that you choose to process, determined by the arguments `start_step`, `clip_length`, and `n_steps`. This is because our initial imitation learning environment only handles single clip imitation, so the clips should be separated during preprocessing.

#### Imitation environment
The environment logic is largely completed (found in `environments.py`). However, we still need to verify its correctness by comparing it with the original dmcontrol tracking environment. Specifically, there are a number of transformations used to convert the expert trajectory data from global frame to a local frame relative to the agent's current state. These transformations were lifted from dmcontrol and there are some bits where we are uncertain about. Thus, some verification is needed.

