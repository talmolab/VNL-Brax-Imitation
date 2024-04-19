"""
This file just contains part of dmcontrol.utils.transformations and surfaces compute_velocity_from_kinematics()
"""

from jax import numpy as jnp

_TOL = 1e-10


def _get_qmat_indices_and_signs():
  """Precomputes index and sign arrays for constructing `qmat` in `quat_mul`."""
  w, x, y, z = range(4)
  qmat_idx_and_sign = jnp.array([
      [w, -x, -y, -z],
      [x, w, -z, y],
      [y, z, w, -x],
      [z, -y, x, w],
  ])
  indices = jnp.abs(qmat_idx_and_sign)
  signs = 2 * (qmat_idx_and_sign >= 0) - 1
  # Prevent array constants from being modified in place. (TODO: jnp never writes in place i think)
  indices.flags.writeable = False
  signs.flags.writeable = False
  return indices, signs


_qmat_idx, _qmat_sign = _get_qmat_indices_and_signs()


def quat_mul(quat1, quat2):
  """Computes the Hamilton product of two quaternions.

  Any number of leading batch dimensions is supported.

  Args:
    quat1: A quaternion [w, i, j, k].
    quat2: A quaternion [w, i, j, k].

  Returns:
    The quaternion product quat1 * quat2.
  """
  # Construct a (..., 4, 4) matrix to multiply with quat2 as shown below.
  qmat = quat1[..., _qmat_idx] * _qmat_sign

  # Compute the batched Hamilton product:
  # |w1 -i1 -j1 -k1|   |w2|   |w1w2 - i1i2 - j1j2 - k1k2|
  # |i1  w1 -k1  j1| . |i2| = |w1i2 + i1w2 + j1k2 - k1j2|
  # |j1  k1  w1 -i1|   |j2|   |w1j2 - i1k2 + j1w2 + k1i2|
  # |k1 -j1  i1  w1|   |k2|   |w1k2 + i1j2 - j1i2 + k1w2|
  return (qmat @ quat2[..., None])[..., 0]


def quat_conj(quat):
  """Return conjugate of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion [w, -i, -j, -k] representing the inverse of the rotation
    defined by `quat` (not assuming normalization).
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = jnp.asarray(quat)
  return jnp.stack(
      [quat[..., 0], -quat[..., 1],
       -quat[..., 2], -quat[..., 3]], axis=-1).astype(jnp.float64)
  
  
def quat_diff(source, target):
  """Computes quaternion difference between source and target quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    source: A quaternion [w, i, j, k].
    target: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the rotation from source to target.
  """
  return quat_mul(quat_conj(source), target)
    

def _clip_within_precision(number, low, high, precision=_TOL):
  """Clips input to provided range, checking precision.

  Args:
    number: (float) number to be clipped.
    low: (float) lower bound.
    high: (float) upper bound.
    precision: (float) tolerance.

  Returns:
    Input clipped to given range.

  Raises:
    ValueError: If number is outside given range by more than given precision.
  """
  if (number < low - precision).any() or (number > high + precision).any():
    raise ValueError(
        'Input {:.12f} not inside range [{:.12f}, {:.12f}] with precision {}'.
        format(number, low, high, precision))

  return jnp.clip(number, low, high)


def quat_to_axisangle(quat):
  """Returns the axis-angle corresponding to the provided quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length.
  """
  angle = 2 * jnp.arccos(_clip_within_precision(quat[0], -1., 1.))

  if angle < _TOL:
    return jnp.zeros(3)
  else:
    qn = jnp.sin(angle/2)
    angle = (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
    axis = quat[1:4] / qn
    return axis * angle

def compute_velocity_from_kinematics(
    qpos_trajectory: jnp.ndarray, dt: float
) -> jnp.ndarray:
    """Computes velocity trajectory from position trajectory.

    Args:
        qpos_trajectory (np.ndarray): trajectory of qpos values T x ?
          Note assumes has freejoint as the first 7 dimensions
        dt (float): timestep between qpos entries

    Returns:
        np.ndarray: Trajectory of velocities.
    """
    qvel_translation = (qpos_trajectory[1:, :3] - qpos_trajectory[:-1, :3]) / dt
    qvel_gyro = []
    for t in range(qpos_trajectory.shape[0] - 1):
        normed_diff = quat_diff(qpos_trajectory[t, 3:7], qpos_trajectory[t + 1, 3:7])
        normed_diff /= jnp.linalg.norm(normed_diff)
        qvel_gyro.append(quat_to_axisangle(normed_diff) / dt)
    qvel_gyro = jnp.stack(qvel_gyro)
    qvel_joints = (qpos_trajectory[1:, 7:] - qpos_trajectory[:-1, 7:]) / dt
    return jnp.concatenate([qvel_translation, qvel_gyro, qvel_joints], axis=1)