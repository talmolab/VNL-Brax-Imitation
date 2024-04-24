import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx


def global_vector_to_local_frame(mjxData, vec_in_world_frame):
    """Linearly transforms a world-frame vector into entity's local frame.

    Note that this function does not perform an affine transformation of the
    vector. In other words, the input vector is assumed to be specified with
    respect to the same origin as this entity's local frame. This function
    can also be applied to matrices whose innermost dimensions are either 2 or
    3. In this case, a matrix with the same leading dimensions is returned
    where the innermost vectors are replaced by their values computed in the
    local frame.
    
    Returns the resulting vector
    """
    xmat = jnp.reshape(mjxData.xmat, (3, 3))
    # The ordering of the np.dot is such that the transformation holds for any
    # matrix whose final dimensions are (2,) or (3,).
    if vec_in_world_frame.shape[-1] == 2:
      return jnp.dot(vec_in_world_frame, xmat[:2, :2])
    elif vec_in_world_frame.shape[-1] == 3:
      return jnp.dot(vec_in_world_frame, xmat)
    else:
      raise ValueError('`vec_in_world_frame` should have shape with final '
                       'dimension 2 or 3: got {}'.format(
                           vec_in_world_frame.shape))

def get_reference_rel_bodies_pos_local(mjxData, clip_reference_features):
    """Observation of the reference bodies relative to walker in local frame."""
    
    # self._walker_features['body_positions'] is the equivalent of 
    # the ref traj 'body_positions' feature but calculated for the current walker state
    time_steps = time_step + ref_steps
    obs = global_vector_to_local_frame(
        mjxData, (clip_reference_features['body_positions'][time_steps] -
                self._walker_features['body_positions'])[:, self._body_idxs])
    return jnp.concatenate([o.flatten() for o in obs])