from flax import linen as nn
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp

from typing import Any, Callable, Sequence, Tuple
import ml_collections
import numpy as np
import optax
import wandb


# Define the KL divergence loss
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class Encoder(nn.Module):
    """
    Encoder that maps the reference encoder to the intention
    latent space. This is for the imitation learning
    """

    intention_size: int = 20
    hidden_layer_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_layer_size, name="fc1")(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.intention_size, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.intention_size, name="fc2_logvar")(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    """VAE Decoder."""

    intention_size: int = 20
    observation_size: int = (
        0  # modify this when we concatenate the intention with observation
    )
    output_size: int = 10

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.intention_size + self.observation_size, name="dec1")(z)
        z = nn.relu(z)
        mean_z = nn.Dense(self.output_size, name="dec2_mean_est")(z)
        return mean_z


class IntentionMapper(nn.Module):
    """Intention Mapper for NMIST classifications"""

    intention_size: int = 20
    hidden_layer_size: int = 128
    output_size: int = 10
    kl_weights: Tuple[float, float] = (0, 0)

    def setup(self):
        self.encoder = Encoder(
            intention_size=self.intention_size, hidden_layer_size=self.hidden_layer_size
        )
        self.decoder = Decoder(
            intention_size=self.intention_size, output_size=self.output_size
        )

    def __call__(self, x, z_rng):
        intention_mean, intention_logvar = self.encoder(x)
        z = reparameterize(z_rng, intention_mean, intention_logvar)
        # fix the variance of the last stochastic layer
        decoder_x_mean = self.decoder(z)
        return decoder_x_mean, intention_mean, intention_logvar