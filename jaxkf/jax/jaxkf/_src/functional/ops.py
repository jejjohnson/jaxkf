from typing import Tuple
import jax
import jax.numpy as jnp
from typing import NamedTuple

from jax.random import multivariate_normal
from tensorflow_probability.substrates import jax as tfp
from einops import rearrange

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


# def forward_filter(x, mu0, Sigma0):

#     return None


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
    mu_t_1 = state.mu_t
    Sigma_t_1 = state.Sigma_t
    t = state.t

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

    state_pred = State(mu_t=mu_z_tt_1, Sigma_t=Sigma_z_tt_1, t=t + 1)

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

    state_corrected = State(mu_t=mu_z_t, Sigma_t=Sigma_z_t, t=t + 1)

    return (state_pred, state_corrected)


def smoother_step():
    pass


# def pred_mean(mu_t, F):
#     """predictive mean for the next time step given the previous.

#     eq: mu_t+1 = mu_t @ F

#     Args:
#         mu_t (Array): the mean from the current time step, t,
#             shape=(*B, D)
#         F (Array): the transition matrix,
#             shape=(D, D)

#     Returns:
#         mu_t1: the mean for the next time step, mu(t+1)
#     """
#     return mu_t @ F


# def pred_cov(Sigma_t, F, Q):
#     """predictive covariance for the next time step given the previous.

#     eq: Sigma_t+1 = F @ Sigma_t @ F.T + Q

#     Args:
#         Sigma_t (Array): the covariance from the current time step, t,
#             shape=(D, D)
#         F (Array): the transition matrix,
#             shape=(D, D)
#         Q (Array): the noise matrix
#             shape=(D, D)

#     Returns:
#         Sigma_t1: the covariance for the next time step, t+1
#     """
#     return F @ Sigma_t @ F.T + Q


# def cond_est_mu(mu_t_cond, K_t, x_t):
#     mu = mu_t_cond + K_t @ x_t
#     return mu


# def cond_est_sigma(K_t, H, Sigma_t_cond):
#     I = jnp.eye(K_t.shape[0])
#     Sigma_t = (I - K_t @ H) @ Sigma_t_cond
#     return Sigma_t


# def compute_innovation_residual(y_t, x_t):
#     """Computes the innovation and innovation covariance

#     Args:
#         y_t ([type]): [description]
#         x_t ([type]): [description]
#         H ([type]): [description]
#         Sigma_t ([type]): [description]
#         R ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     innovation = y_t - x_t

#     return innovation


# def compute_innovation_cov(H, Sigma_t, R):
#     return H @ Sigma_t @ H.T + R


# def kalman_gain(innovation_cov, H, Sigma_t_cond):
#     """Computes the Kalman gain from the innovation and obs model

#     Args:
#         innovation_cov ([type]): [description]
#         H ([type]): [description]
#         Sigma_t_cond ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     R = Sigma_t_cond @ H.T @ jnp.linalg.inv(innovation_cov)

#     return R
