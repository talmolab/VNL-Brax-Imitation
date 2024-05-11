import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.networks import MLP
import brax.training.agents.ppo.networks as ppo_networks
from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn


class Encoder(nn.Module):
    '''outputs in the form of distributions in latent space'''
    layer_sizes: Sequence[int]
    latents: int  # intention size
    activation: networks.ActivationFn = nn.tanh
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # For each layer in the sequence
        # Make a dense net and apply layernorm then tanh
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = nn.LayerNorm(x)
            x = self.activation(x)

        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    '''decode with action output'''
    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.tanh
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
        return x


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class IntentionNetwork(nn.Module):
    """Full VAE model, encode -> decode with sampled actions"""

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latents: int = 60

    def setup(self):
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(self, traj, obs):
        encoder_rng, decoder_rng = self.make_rng("encoder"), self.make_rng("decoder")
        # construct the intention network
        intention_mean, intention_logvar = self.encoder(traj)
        z = reparameterize(encoder_rng, intention_mean, intention_logvar)
        action_mean = self.decoder(z.cat(obs))
        action_sample = reparameterize(
            decoder_rng, action_mean, jnp.ones_like(action_mean)
        )
        return action_sample


def make_intention_policy(
    param_size: int,
    latent_size: int,
    obs_size: int,
    traj_size: int,  # the size of the intended trajectory
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_layer_sizes: Sequence[int] = (1024, 1024),
) -> IntentionNetwork:
    """Creates an intention policy network."""

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_layer_sizes) + [latent_size],
        decoder_layers=list(decoder_layer_sizes) + [param_size],
        latents=param_size,
    )

    def apply(processor_params, policy_params, traj, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, traj=traj, obs=obs)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_traj = jnp.zeros((1, traj_size))

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_traj), apply=apply
    )
