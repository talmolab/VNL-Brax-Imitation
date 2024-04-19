"""Network definitions."""
# making it a inherent class later！

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.networks import MLP

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]

  
class Encoder(nn.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    latents: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # For each layer in the sequence
        # Make a dense net and apply layernorm then tanh
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(x)
            x = nn.LayerNorm(x)
            x = self.activation(x)
            
        mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
        return mean_x, logvar_x
    
class Decoder(nn.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
        return x

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

class VAE(nn.Module):
  """Full VAE model."""

  encoder_layers: Sequence[int]
  decoder_layers: Sequence[int]
  latents: int = 60

  def setup(self):
    self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
    self.decoder = Decoder(layer_sizes=self.decoder_layers)

  # x = (traj_dims * traj_length + state_dims)
  def __call__(self, x, e_rng, z_rng):
    traj = x[:traj_dims * traj_length]
    state = x[traj_dims * traj_length:]
    mean, logvar = self.encoder(traj, e_rng)
    z = reparameterize(z_rng, mean, logvar)
    action = self.decoder(z.cat(state))
    return action, mean, logvar

  def generate(self, z):
    return self.decoder(z) + noise


@flax.struct.dataclass
class PPONetworks:
  policy_network: VAE
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution
  
  
def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = policy_network.apply(*params, observations)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample)
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions)
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions
      }

    return policy

  return make_policy

  
def make_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    encoder_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_layer_sizes: Sequence[int] = (1024),
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    ) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_network = make_policy_vae(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      encoder_layer_sizes=encoder_layer_sizes,
      decoder_layer_sizes=decoder_layer_sizes,
      )
  value_network = make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution)
  
  
def make_policy_vae(
    param_size: int,
    latent_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    encoder_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_layer_sizes: Sequence[int] = (1024),
    ) -> VAE:
  """Creates a policy network."""
  
  policy_module = VAE(
      encoder_layers=list(encoder_layer_sizes) + [latent_size],
      decoder_layers=list(decoder_layer_sizes) + [param_size], 
      latents = param_size)

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_network(
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu) -> FeedForwardNetwork:
  """Creates a policy network."""

  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)

