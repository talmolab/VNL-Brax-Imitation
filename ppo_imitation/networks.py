import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
import brax.training.agents.ppo.networks as ppo_networks
from brax.training.types import PRNGKey
from brax.training.networks import MLP

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class ImitationMLP(nn.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.he_normal()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = True

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            if i == len(self.layer_sizes) - 1:
                self.kernel_init = jax.nn.initializers.zeros
            hidden = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
                if self.layer_norm:
                    hidden = nn.LayerNorm()(hidden)
        return hidden


def make_mlp_policy(
    param_size: int,
    obs_size: int,
    traj_size: int,  # the size of the intended trajectory
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    layer_sizes: Sequence[int] = (256,) * 2,
):
    """Creates an intention policy network."""

    policy_module = ImitationMLP(layer_sizes=list(layer_sizes) + [param_size])

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, data=obs)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_traj = jnp.zeros((1, traj_size))

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(
            key, data=jnp.concatenate([dummy_traj, dummy_obs], axis=-1)
        ),
        apply=apply,
    )


# add traj size so value takes that as well
def make_value_network(
    traj_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation=nn.relu,
) -> networks.FeedForwardNetwork:
    """Creates a policy network."""
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_traj = jnp.zeros((1, traj_size))

    return networks.FeedForwardNetwork(
        init=lambda key: value_module.init(
            key, jnp.concatenate([dummy_traj, dummy_obs], axis=-1)
        ),
        apply=apply,
    )
