"""
Custom network definitions.
This is needed because we need to route the observations 
to proper places in the network in the case of the VAE (CoMic, Hasenclever 2020)
"""

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.networks import MLP

from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

from ppo_imitation import intention_policy_network as ipn


@flax.struct.dataclass
class PPOImitationNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPOImitationNetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        # can modify this to provide stochastic action + noise
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            trajectories: types.Observation,
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            logits, _, _ = policy_network.apply(
                *params, trajectories, observations, key_network
            )
            # logits comes from policy directly, raw predictions that decoder generates (action, intention_mean, intention_logvar)

            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}

            # action sampling is happening here, according to distribution parameter logits
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )

            # probability of selection specific action, actions with higher reward should have higher probability
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)

            action_size = logits.shape[-1] // 2
            random_actions = jax.random.uniform(
                key_sample, shape=(action_size,), minval=-1, maxval=1
            )
            rand_log_prob = parametric_action_distribution.log_prob(
                logits, random_actions
            )

            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                "log_prob": log_prob,
                "rand_log_prob": rand_log_prob,  # should be low
                "raw_action": raw_actions,
                "logits": logits,  # logits is previous raw action, mean, sd
            }

        return policy

    return make_policy


# intention policy
def make_intention_ppo_networks(
    traj_size: int,
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 64,
    encoder_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPOImitationNetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = ipn.make_intention_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        traj_size=traj_size,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_layer_sizes=encoder_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
