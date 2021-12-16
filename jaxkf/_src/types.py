from typing import NamedTuple
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class KFParams(NamedTuple):
    transition_matrix: jnp.ndarray
    transition_noise: jnp.ndarray
    observation_matrix: jnp.ndarray
    observation_noise: jnp.ndarray


class KFParamsDist(NamedTuple):
    transition_matrix: jnp.ndarray
    transition_noise: jnp.ndarray
    transition_noise_dist: tfd.Distribution
    observation_matrix: jnp.ndarray
    observation_noise: jnp.ndarray
    observation_noise_dist: tfd.Distribution

    @property
    def transition_noise_mean(self):
        return self.transition_noise_dist.mean()

    @property
    def transition_noise_covariance(self):
        return self.transition_noise_dist.covariance()

    @property
    def observation_noise_mean(self):
        return self.observation_noise_dist.mean()

    @property
    def observation_noise_covariance(self):
        return self.observation_noise_dist.covariance()


class State(NamedTuple):
    mu_t: jnp.ndarray
    Sigma_t: jnp.ndarray
    t: int


class StatePrior(NamedTuple):
    prior: tfd.Distribution
    mu0: jnp.ndarray
    Sigma0: jnp.ndarray

    @property
    def mean(self):
        return self.prior.mean()

    @property
    def covariance(self):
        return self.prior.covariance()

    def sample(self, **kwargs):
        return self.prior.sample(**kwargs)


class FilteredState(NamedTuple):
    mu_filtered: jnp.ndarray
    Sigma_filtered: jnp.ndarray
    log_likelihoods: jnp.ndarray
    mu_cond: jnp.ndarray
    Sigma_cond: jnp.ndarray
    ts: jnp.ndarray
