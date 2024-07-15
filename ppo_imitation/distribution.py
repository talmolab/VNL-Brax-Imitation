"""Brax's Probability distributions in JAX. Modified for VNL-Brax_Imitation"""

import abc
import jax
import jax.numpy as jnp


class ParametricDistribution(abc.ABC):
    """Abstract class for parametric (action) distribution."""

    def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
        """Abstract class for parametric (action) distribution.

        Specifies how to transform distribution parameters (i.e. actor output)
        into a distribution over actions.

        Args:
          param_size: size of the parameters for the distribution
          postprocessor: bijector which is applied after sampling (in practice, it's
            tanh or identity)
          event_ndims: rank of the distribution sample (i.e. action)
          reparametrizable: is the distribution reparametrizable
        """
        self._param_size = param_size
        self._postprocessor = postprocessor
        self._event_ndims = event_ndims  # rank of events
        self._reparametrizable = reparametrizable
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters):
        """Creates distribution from parameters."""
        pass

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._reparametrizable

    def postprocess(self, event):
        return self._postprocessor.forward(event)

    def inverse_postprocess(self, event):
        return self._postprocessor.inverse(event)

    def sample_no_postprocessing(self, parameters, seed):
        return self.create_dist(parameters).sample(seed=seed)

    def sample(self, parameters, seed):
        """Returns a sample from the postprocessed distribution."""
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def mode(self, parameters):
        """Returns the mode of the postprocessed distribution."""
        return self.postprocess(self.create_dist(parameters).mode())

    def log_prob(self, parameters, actions):
        """Compute the log probability of actions."""
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(actions)
        log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
        return log_probs

    def entropy(self, parameters, seed):
        """Return the entropy of the given distribution."""
        dist = self.create_dist(parameters)
        entropy = dist.entropy()
        entropy += self._postprocessor.forward_log_det_jacobian(dist.sample(seed=seed))
        if self._event_ndims == 1:
            entropy = jnp.sum(entropy, axis=-1)
        return entropy


class NormalDistribution:
    """Normal distribution."""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, seed):
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * jnp.ones_like(self.loc)


class TanhBijector:
    """Tanh Bijector."""

    def forward(self, x):
        return jnp.tanh(x)

    def inverse(self, y):
        return jnp.arctanh(y)

    def forward_log_det_jacobian(self, x):
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))


class ActionClipping:
    """
    Clip the output of the NN to be within the range of -1 and 1
    """

    def forward(self, x):
        return jnp.clip(x, -1, 1) # clipping it to -0.5 to 0.5 and scale the output of the x by the factor of 5

    def forward_log_det_jacobian(self, x):
        # since this is a linear transformations.
        return 0.0


class NormalTanhDistribution(ParametricDistribution):
    """Normal distribution followed by tanh."""

    def __init__(self, event_size, min_std=0.001, var_scale=1):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.
          var_scale: adjust the gaussian's scale parameter.
        """
        # We apply tanh to gaussian actions to bound them.
        # Normally we would use TransformedDistribution to automatically
        # apply tanh to the distribution.
        # We can't do it here because of tanh saturation
        # which would make log_prob computations impossible. Instead, most
        # of the code operate on pre-tanh actions and we take the postprocessor
        # jacobian into account in log_prob computations.
        super().__init__(
            param_size=2 * event_size,
            postprocessor=TanhBijector(),
            event_ndims=1,
            reparametrizable=True,
        )
        self._min_std = min_std
        self._var_scale = var_scale

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        return NormalDistribution(loc=loc, scale=scale)


class NormalTanhDistributionFixedStd(ParametricDistribution):
    """Normal distribution with fixed variance followed by tanh."""

    def __init__(self, event_size, scale=0.1):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.
          var_scale: adjust the gaussian's scale parameter.
        """
        # We apply tanh to gaussian actions to bound them.
        # Normally we would use TransformedDistribution to automatically
        # apply tanh to the distribution.
        # We can't do it here because of tanh saturation
        # which would make log_prob computations impossible. Instead, most
        # of the code operate on pre-tanh actions and we take the postprocessor
        # jacobian into account in log_prob computations.
        super().__init__(
            param_size=2 * event_size,
            postprocessor=TanhBijector(),
            event_ndims=1,
            reparametrizable=True,
        )
        self._scale = scale

    def create_dist(self, parameters):
        scale = jnp.broadcast_to(self._scale, parameters.shape)
        return NormalDistribution(loc=parameters, scale=scale)


class NormalClipDistributionFixedStd(ParametricDistribution):
    """Normal distribution with fixed variance followed by Clip output range."""

    def __init__(self, event_size, scale=0.1):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.
          var_scale: adjust the gaussian's scale parameter.
        """
        # We apply tanh to gaussian actions to bound them.
        # Normally we would use TransformedDistribution to automatically
        # apply tanh to the distribution.
        # We can't do it here because of tanh saturation
        # which would make log_prob computations impossible. Instead, most
        # of the code operate on pre-tanh actions and we take the postprocessor
        # jacobian into account in log_prob computations.
        super().__init__(
            param_size=2 * event_size,
            postprocessor=ActionClipping(),
            event_ndims=1,
            reparametrizable=False,
        )
        self._scale = scale

    def create_dist(self, parameters):
        scale = jnp.broadcast_to(self._scale, parameters.shape)
        return NormalDistribution(loc=parameters, scale=scale)
