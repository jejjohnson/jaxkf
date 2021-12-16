import jax.numpy as jnp
from jaxkf._src.models.kalmanfilter import init_kf_params


def make_noisy_signal_model(
    prior_noise: float = 1e-4,
    trans_noise: float = 1e-2,
    obs_noise: float = 1e-2,
):
    # initialize dims
    state_dim = 2
    obs_dim = 1

    # ======================
    # state prior
    # ======================

    # init prior params
    mu0 = jnp.ones(state_dim)
    Sigma0 = prior_noise * jnp.ones(state_dim)

    assert mu0.shape == (state_dim,)
    assert Sigma0.shape == (state_dim,)

    # ======================
    # transition model
    # ======================

    # state transition matrix
    transition_matrix = jnp.array([[1.0, 1.0], [0.0, 1.0]])

    # transition uncertainty
    transition_noise = trans_noise * jnp.ones((state_dim))

    # ======================
    # emission model
    # ======================

    # observation matrix
    observation_matrix = jnp.array([[1.0, 0.0]])

    # observation uncertainty
    observation_noise = obs_noise * jnp.ones((obs_dim))

    # ======================
    # INIT MODEL
    # ======================
    params_dist, state_prior = init_kf_params(
        mu0=mu0,
        Sigma0=Sigma0,
        transition_matrix=transition_matrix,
        transition_noise=transition_noise,
        observation_matrix=observation_matrix,
        observation_noise=observation_noise,
    )

    return params_dist, state_prior
