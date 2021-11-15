from typing import Tuple
import jax
import chex

Array = chex.Array


def __forward_filter(self, x: Array) -> Tuple[Array, Array, Array, Array]:
    mu0 = self.initial_state_prior.mean()
    Sigma0 = self.initial_state_prior.covariance()
    _, (
        log_likelihoods,
        filtered_means,
        filtered_covs,
        mu_cond_hist,
        Sigma_cond_hist,
    ) = jax.lax.scan(self.__kalman_step, (mu0, Sigma0), x)

    return log_likelihoods, filtered_means, filtered_covs, mu_cond_hist, Sigma_cond_hist


def forward_filter(x: Array) -> Tuple[Array, Array, Array, Array]:
    """
    Run a Kalman filter over a provided sequence of outputs.

    Parameters
    ----------
    x_hist: array(*batch_size, timesteps, observation_size)

    Returns
    -------
    * array(*batch_size, timesteps, state_size):
        Filtered means mut
    * array(*batch_size, timesteps, state_size, state_size)
        Filtered covariances Sigmat
    * array(*batch_size, timesteps, state_size)
        Filtered conditional means mut|t-1
    * array(*batch_size, timesteps, state_size, state_size)
        Filtered conditional covariances Sigmat|t-1
    """
    forward_map = jax.vmap(self.__forward_filter, 0)

    *batch_shape, timesteps, _ = x.shape
    state_mean_dims = (*batch_shape, timesteps, self.state_size)
    state_cov_dims = (*batch_shape, timesteps, self.state_size, self.state_size)

    x = x.reshape(-1, timesteps, self.observation_size)
    (
        log_likelihoods,
        filtered_means,
        filtered_covs,
        mu_cond_hist,
        Sigma_cond_hist,
    ) = forward_map(x)

    log_likelihoods = log_likelihoods.reshape(*batch_shape, timesteps)
    filtered_means = filtered_means.reshape(state_mean_dims)
    filtered_covs = filtered_covs.reshape(state_cov_dims)
    mu_cond_hist = mu_cond_hist.reshape(state_mean_dims)
    Sigma_cond_hist = Sigma_cond_hist.reshape(state_cov_dims)

    return log_likelihoods, filtered_means, filtered_covs, mu_cond_hist, Sigma_cond_hist
