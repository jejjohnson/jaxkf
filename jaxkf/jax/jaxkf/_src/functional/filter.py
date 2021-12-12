from typing import Sequence, Tuple, Union
import collections.abc
import operator
import functools
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from tensorflow_probability.substrates import jax as tfp
from einops import rearrange, repeat

IntLike = Union[int, np.int16, np.int32, np.int64]

tfd = tfp.distributions


class KFParams(NamedTuple):
    transition_matrix: jnp.ndarray
    transition_noise: jnp.ndarray
    observation_matrix: jnp.ndarray
    observation_noise: jnp.ndarray


class KFParamsDist(NamedTuple):
    transition_matrix: jnp.ndarray
    transition_noise_dist: tfd.Distribution
    observation_matrix: jnp.ndarray
    observation_noise_dist: tfd.Distribution


class State(NamedTuple):
    mu_t: jnp.ndarray
    Sigma_t: jnp.ndarray
    t: int


def filter_step_sequential(obs, state_init, params):

    # initialize state
    state_init = (
        state_init.prior.mean(),
        state_init.prior.covariance(),
        0,
    )

    # define ad-hoc body for kalman step
    def body(state, obs):

        # do Kalman Step
        state = kalman_step(obs, state, params)

        # unroll
        mu_z_t, Sigma_z_t, mu_z_tt_1, Sigma_z_tt_1, t = state

        # calculate log probability
        log_prob = filter_log_prob(obs, mu_z_t, Sigma_z_t, params)

        state = (mu_z_t, Sigma_z_t, t)

        return state, (mu_z_t, Sigma_z_t, log_prob, mu_z_tt_1, Sigma_z_tt_1, t)

    # loop through samples
    _, states = jax.lax.scan(body, state_init, obs)

    return states[0], states[1], states[2], states[3], states[4], states[5].squeeze()


def forward_filter(obs, state_init, params: KFParams):

    # define forward map
    fn = lambda obs: filter_step_sequential(obs, state_init, params)

    vmap_fn = jax.vmap(fn, in_axes=0)

    # handle dimensions
    state_dims = params.transition_matrix.shape[0]
    obs_dims = params.observation_matrix.shape[0]

    *batch_shape, timesteps, _ = obs.shape
    state_mean_dims = (*batch_shape, timesteps, state_dims)
    state_cov_dims = (*batch_shape, timesteps, state_dims, state_dims)

    obs = obs.reshape(-1, timesteps, obs_dims)

    # forward filter
    (
        filtered_means,
        filtered_covs,
        log_likelihoods,
        mu_cond_hist,
        Sigma_cond_hist,
        ts,
    ) = vmap_fn(obs)

    # reshape to appropriate dimensions
    filtered_means = filtered_means.reshape(state_mean_dims)
    filtered_covs = filtered_covs.reshape(state_cov_dims)
    log_likelihoods = log_likelihoods.reshape(*batch_shape, timesteps)
    mu_cond_hist = mu_cond_hist.reshape(state_mean_dims)
    Sigma_cond_hist = Sigma_cond_hist.reshape(state_cov_dims)

    return (
        filtered_means,
        filtered_covs,
        log_likelihoods,
        mu_cond_hist,
        Sigma_cond_hist,
        ts.squeeze(),
    )


def kalman_step(x_t, state: State, params: KFParams):
    """Computes the c

    Args:
        x_t ([type]): [description]
        state ([type]): [description]
        params ([type]): [description]

    Returns:
        [type]: [description]
    """

    # unroll state
    mu_t_1, Sigma_t_1, t = state

    # unroll KF params
    F = params.transition_matrix
    Q = params.transition_noise
    H = params.observation_matrix
    R = params.observation_noise

    # get dims
    state_dim = F.shape[-1]
    I = jnp.eye(state_dim)

    # ==============
    # PREDICT STEP
    # ==============

    # predictive mean, cov (state), t|t-1
    mu_z_tt_1 = F @ mu_t_1
    Sigma_z_tt_1 = F @ Sigma_t_1 @ F.T + Q

    # predictive mean, cov (obs), t
    mu_x_t = H @ mu_z_tt_1
    Sigma_x_t = H @ Sigma_z_tt_1 @ H.T + R

    # ============
    # UPDATE STEP
    # ============

    # innovation
    r_t = x_t - mu_x_t

    # Kalman gain
    K_t = Sigma_z_tt_1 @ H.T @ jnp.linalg.inv(Sigma_x_t)

    # correction
    mu_z_t = mu_z_tt_1 + K_t @ r_t
    Sigma_z_t = (I - K_t @ H) @ Sigma_z_tt_1

    return mu_z_t, Sigma_z_t, mu_z_tt_1, Sigma_z_tt_1, t + 1


def filter_log_prob(x_t, mu_z, Sigma_z, params):

    # unroll KF params
    F = params.transition_matrix
    Q = params.transition_noise
    H = params.observation_matrix
    R = params.observation_noise

    # predict step (state)
    mu_z_t, Sigma_z_t = predict_step_obs(mu_z, Sigma_z, F, Q)

    # predict step (obs)
    mu_x_t, Sigma_x_t = predict_step_obs(mu_z_t, Sigma_z_t, H, R)

    dist = tfd.MultivariateNormalFullCovariance(mu_x_t, Sigma_x_t)

    log_likelihood = dist.log_prob(x_t)

    return log_likelihood


def predict_step_state(mu, Sigma, F, Q):
    # predictive mean, cov (state), t|t-1
    mu_z_t = F @ mu
    Sigma_z_t = F @ Sigma @ F.T + Q

    return mu_z_t, Sigma_z_t


def predict_step_obs(mu_z_t, Sigma_z_t, H, R):
    # predictive mean, cov (obs), t
    mu_x_t = H @ mu_z_t
    Sigma_x_t = H @ Sigma_z_t @ H.T + R

    return mu_x_t, Sigma_x_t


def update_step(mu_z_t, Sigma_z_t, mu_x_t, Sigma_x_t, x_t, H):
    # innovation
    r_t = x_t - mu_x_t

    # Kalman gain
    K_t = Sigma_z_t @ H.T @ jnp.linalg.inv(Sigma_x_t)

    I = jnp.eye(Sigma_z_t.shape[-1])

    # correction
    mu_z_t = mu_z_t + K_t @ r_t
    Sigma_z_t = (I - K_t @ H) @ Sigma_z_t

    return mu_z_t, Sigma_z_t
