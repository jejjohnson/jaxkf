import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from jaxkf._src.models.lgssm import LGSSM
import treex as tx


class KalmanFilter(LGSSM):
    transition_matrix: tx.State.node()
    transition_noise: tx.Paramter.node()
    observation_matrix: tx.State.node()
    observation_noise: tx.State.node()


def init_kf_params(
    mu0,
    Sigma0,
    transition_matrix,
    transition_noise,
    observation_matrix,
    observation_noise,
):
    # check dimensions
    assert mu0.ndim == 1
    assert Sigma0.ndim == 1
    assert transition_matrix.ndim == 2
    assert transition_noise.ndim == 1
    assert observation_matrix.ndim == 2
    assert observation_noise.ndim == 1

    # check all matching dims
    state_dim = mu0.shape[0]
    obs_dim = observation_noise.shape[0]

    # =================
    # TRANSITION MODEL
    # =================

    # create trans noise dist
    transition_noise_dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(state_dim), scale_diag=transition_noise
    )

    # check sizes
    assert transition_matrix.shape == (state_dim, state_dim)
    assert transition_noise.shape == (state_dim,)
    assert transition_noise_dist.mean().shape == (state_dim,)
    assert transition_noise_dist.covariance().shape == (state_dim, state_dim)

    # ================
    # EMISSION MODEL
    # ================

    # create emission noise dist
    observation_noise_dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(obs_dim), scale_diag=observation_noise
    )

    # check dims for observation matrix
    assert observation_matrix.shape == (obs_dim, state_dim)
    assert observation_noise.shape == (obs_dim,)
    assert observation_noise_dist.mean().shape == (obs_dim,)
    assert observation_noise_dist.covariance().shape == (obs_dim, obs_dim)

    # ======================
    # state space model
    # ======================

    # init kf params (dist)
    params_dist = KFParamsDist(
        transition_matrix=transition_matrix,
        transition_noise=transition_noise,
        transition_noise_dist=transition_noise_dist,
        observation_matrix=observation_matrix,
        observation_noise=observation_noise,
        observation_noise_dist=observation_noise_dist,
    )

    # init prior dist
    prior_dist = tfd.MultivariateNormalDiag(loc=mu0, scale_diag=Sigma0)
    state_prior = StatePrior(mu0=mu0, Sigma0=Sigma0, prior=prior_dist)

    return params_dist, state_prior
