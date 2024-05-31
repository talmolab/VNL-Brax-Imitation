import jax
from .intention_coder import *
import optax
import wandb

# this is a record of the way that we invoke the intention coder in the training loop
# TODO: we need to adapt this to the ppo algorithm and modify how they interact with the intention coder.

@jax.jit
def apply_model(state, images, labels, kl_weights, rng):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params, rng):
        # added stochastic layer at loss fn
        apply_rng, last_sto_rng = jax.random.split(rng)
        decoder_x_mean, intention_mean, intention_logvar = state.apply_fn(
            {"params": params}, images, apply_rng
        )
        # Calculate the loss function for the model
        # with regularization terms of the intermediate layers
        decoder_x_logvar = jnp.ones((decoder_x_mean.shape))
        # KL divergence loss for the encoder
        kl_loss_encoder = kl_divergence(intention_mean, intention_logvar)
        # jax.debug.print(
        #     "intention_mean: {}, intention_logvar: {}", intention_mean, intention_logvar
        # )
        # jax.debug.print("kl_loss_encoder: {}", kl_loss_encoder)
        # KL divergence loss for the decoder
        kl_loss_decoder = kl_divergence(decoder_x_mean, decoder_x_logvar)
        logits = reparameterize(last_sto_rng, decoder_x_mean, decoder_x_logvar)
        one_hot = jax.nn.one_hot(labels, 10)
        cross_entro_loss = jnp.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        )
        # Total loss
        kl_weight_encoder, kl_weight_decoder = kl_weights  # unpack kl_weights
        # jax.debug.print(
        #     "kl_loss_encoder: {}, weight: {}", kl_loss_encoder, kl_weight_encoder
        # )
        total_loss = (
            cross_entro_loss
            + kl_weight_encoder * kl_loss_encoder  # hard coded kl weight for now
            + kl_weight_decoder * kl_loss_decoder
        ).squeeze()  # the loss should be a single term # is mean the correct way?
        # total_loss = cross_entro_loss # DEBUGGING
        return total_loss, (logits, kl_loss_encoder, kl_loss_decoder, cross_entro_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)
    (loss, (logits, kl_loss_encoder, kl_loss_decoder, cross_entro_loss)), grads = (
        grad_fn(state.params, rng)
    )  # If has_aux is True then a tuple of ((value, auxiliary_data), gradient) is returned.
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, kl_loss_encoder, kl_loss_decoder, cross_entro_loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, config, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...].reshape((batch_size, 784))
        batch_labels = train_ds["label"][perm, ...]
        kl_weights = (config.kl_weight_encoder, config.kl_weight_decoder)
        grads, loss, kl_loss_encoder, kl_loss_decoder, cross_entro_loss, accuracy = (
            apply_model(state, batch_images, batch_labels, kl_weights, rng)
        )
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return (
        state,
        train_loss,
        train_accuracy,
        kl_loss_encoder,
        kl_loss_decoder,
        cross_entro_loss,
    )


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    model_key, z_key = jax.random.split(rng)
    kl_weights = (config.kl_weight_encoder, config.kl_weight_decoder)
    mapper = IntentionMapper(
        intention_size=20, hidden_layer_size=128, output_size=10, kl_weights=kl_weights
    )
    params = mapper.init(model_key, jnp.ones([1, 784]), z_key)["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=mapper.apply, params=params, tx=tx)


def create_logger(name, project, config, notes=None):
    return wandb.init(name=name, project=project, config=config, notes=notes)


def train_and_evaluate(
    config: ml_collections.ConfigDict, train_ds, test_ds
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    rng = jax.random.key(0)

    run = create_logger(
        f"kl-we(en,de): ({config.kl_weight_encoder}, {config.kl_weight_decoder})",
        "flax_train",
        config.to_dict(),
        "Experiment with different KL weights",
    )

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        (
            state,
            train_loss,
            train_accuracy,
            kl_loss_encoder,
            kl_loss_decoder,
            cross_entro_loss,
        ) = train_epoch(state, train_ds, config.batch_size, config, input_rng)
        rng, input_rng = jax.random.split(rng)
        kl_weights = (config.kl_weight_encoder, config.kl_weight_decoder)
        _, test_loss, _, _, _, test_accuracy = apply_model(
            state,
            test_ds["image"].reshape((-1, 784)),
            test_ds["label"],
            kl_weights,
            input_rng,
        )
        # log the performance
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "kl_loss_encoder": kl_loss_encoder,
                "kl_loss_decoder": kl_loss_decoder,
                "cross_entro_loss": cross_entro_loss,
            }
        )
    return state
