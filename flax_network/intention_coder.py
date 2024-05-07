from flax import linen as nn
from flax.training import train_state
import jax
from jax import random 
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import wandb


class Encoder(nn.Module):
  """
  Encoder that maps the sensory input to the intention
  latent space.
  """

  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(500, name='fc1')(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
    return mean_x, logvar_x


class Decoder(nn.Module):
  """VAE Decoder."""

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(15, name='dec1')(z)
    z = nn.relu(z)
    mean_z = nn.Dense(10, name='dec2_mean_est')(z)
    return mean_z


class IntentionMapper(nn.Module):
  """Intention Mapper for NMIST classifications"""

  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x #, mean, logvar


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


def model(latents):
  return IntentionMapper(latents=latents)

@jax.jit
def apply_model(state, images, labels, rng):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params, rng):
    # added stochastic layer at loss fn
    apply_rng, last_sto_rng = jax.random.split(rng)
    stochastic_mean = state.apply_fn({'params': params}, images, apply_rng)
    logits = reparameterize(last_sto_rng, stochastic_mean, jnp.ones((stochastic_mean.shape)))
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params, rng)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...].reshape((batch_size, 784))
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels, rng)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy

def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  model_key, z_key = jax.random.split(rng)
  mapper = model(20)
  params = mapper.init(model_key, jnp.ones([1, 784]), z_key)['params']
  tx = optax.sgd(config.learning_rate, config.momentum)
  return train_state.TrainState.create(apply_fn=mapper.apply, params=params, tx=tx)

def create_logger(name, project, config, notes=None):
  return wandb.init(name=name, project=project, config=config, notes=notes)

def train_and_evaluate(
    config: ml_collections.ConfigDict
) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """
  train_ds, test_ds = get_datasets()
  rng = jax.random.key(0)

  run = create_logger("flax_test", "flax_train", config.to_dict(), "added last layer of stochasicity")

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)

  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, config.batch_size, input_rng
    )
    rng, input_rng = jax.random.split(rng)
    _, test_loss, test_accuracy = apply_model(
        state, test_ds['image'].reshape((-1, 784)), test_ds['label'], input_rng,
    )
    # log the performance
    wandb.log({
      'epoch': epoch,
      'train_loss': train_loss,
      'train_accuracy': train_accuracy,
      'test_loss': test_loss,
      'test_accuracy': test_accuracy,
    })
  return state