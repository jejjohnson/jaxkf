# TODO: Do sample for event dimensions

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


class KFParamsDist(NamedTuple):
    transition_matrix: jnp.ndarray
    transition_noise_dist: tfd.Distribution
    observation_matrix: jnp.ndarray
    observation_noise_dist: tfd.Distribution


def sample_sequential(
    prior: tfd.Distribution,
    params: KFParamsDist,
    key,
    num_time_steps: int = 1,
    sample_prior: bool = False,
):
    """Does sequential sampling with the scan function.

    Args:
        state_init ([type]): [description]
        params (KFParamsDist): [description]
        key ([type]): [description]
        num_time_steps (int, optional): [description]. Defaults to 1.
        sample_prior (bool): whether to sample the prior or not.

    Returns:
        [type]: [description]
    """
    # initialize keys for steps
    key_init, key_step = jax.random.split(key, 2)

    # initialize state mean
    if sample_prior:
        state_init = prior.sample(seed=key_init)
    else:
        state_init = prior.mean()

    # initialize steps
    key_steps = jax.random.split(key_step, num_time_steps)

    # sample function
    def body(state, key):

        state, obs = sample_step(state, params, key)

        return state, (state, obs)

    # create ad-hoc func
    fn = lambda states, keys: jax.lax.scan(body, states, keys)

    # vmap_fn = jax.vmap(fn)
    # do loop scan
    _, (state_samples, obs_samples) = fn(state_init, key_steps)

    return state_samples, obs_samples


def sample_sequential_vectorized(
    prior: tfd.Distribution,
    params: KFParamsDist,
    seed,
    num_samples: int = 1,
    num_time_steps: int = 10,
    sample_prior: bool = False,
):
    """[summary]

    Args:
        prior (tfd.Distribution): [description]
        params (KFParamsDist): [description]
        key ([type]): [description]
        num_samples (int, optional): [description]. Defaults to 1.
        num_time_steps (int, optional): [description]. Defaults to 10.
        sample_prior (bool, optional): whether to sample the prior or not.
            Defaults to False

    Returns:
        [type]: [description]
    """
    # initialize keys for steps
    if seed is None:
        key = jax.random.PRNGKey(123)
    if isinstance(seed, int):
        key = jax.random.PRNGKey(seed)

    key_init, key_steps = jax.random.split(key, 2)

    # initialize states
    state_dims = prior.mean().shape[0]

    if sample_prior:
        states_init = prior.sample(seed=key_init, sample_shape=(num_samples,))
    else:
        states_init = repeat(prior.mean(), "D -> B D", B=num_samples, D=state_dims)

    # initialize keys for samples
    key_samples = jax.random.split(key_steps, num_samples * num_time_steps)
    key_samples = rearrange(
        key_samples, "(B T) D -> B T D", B=num_samples, T=num_time_steps, D=state_dims
    )

    # sample body function
    def body(state, key):

        state, obs = sample_step(state, params, key)

        return state, (state, obs)

    # create ad-hoc func
    fn = lambda states, keys: jax.lax.scan(body, states, keys)

    # vectorized mapping
    vmap_fn = jax.vmap(fn)

    # generate samples
    _, (state_samples, obs_samples) = vmap_fn(states_init, key_samples)

    return state_samples, obs_samples


def sample_step(state_t: jnp.ndarray, params: KFParamsDist, key):
    # create keys for noise sampling
    key_trans, key_obs = jax.random.split(key, 2)

    # unroll params
    F = params.transition_matrix
    Q = params.transition_noise_dist.sample(seed=key_trans)
    H = params.observation_matrix
    R = params.observation_noise_dist.sample(seed=key_obs)

    # transition model
    state_t_t1 = F @ state_t + Q

    # emission model
    obs_mu_t = H @ state_t_t1 + R

    return state_t_t1, obs_mu_t


def sample_event(
    prior,
    params,
    seed,
    num_time_steps: int,
    sample_shape: Union[IntLike, Sequence[IntLike]] = (),
    sample_prior: bool = False,
):
    """Samples an event.
    Parameters
    ----------
    seed: PRNG key or integer seed.
    num_timesteps: int
    sample_shape: Additional leading dimensions for sample.
    Returns
    -------
    * Array(*sample_shape, batch_shape, event_shape)
        A sample of shape `sample_shape` + `batch_shape` + `event_shape`.
    """

    rng, sample_shape = convert_seed_and_sample_shape(seed, sample_shape)
    num_samples = functools.reduce(operator.mul, sample_shape, 1)  # product

    state_samples, obs_samples = sample_sequential_vectorized(
        prior,
        params,
        seed=rng,
        num_samples=num_samples,
        num_time_steps=num_time_steps,
        sample_prior=sample_prior,
    )

    state_samples = state_samples.reshape(sample_shape + state_samples.shape[1:])
    obs_samples = obs_samples.reshape(sample_shape + obs_samples.shape[1:])

    return state_samples, obs_samples


def convert_seed_and_sample_shape(
    seed: Union[IntLike, jax.random.PRNGKey],
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Tuple[jax.random.PRNGKey, Tuple[int, ...]]:
    """Shared functionality to ensure that seeds and shapes are the right type."""
    if not isinstance(sample_shape, collections.abc.Sequence):
        sample_shape = (sample_shape,)
        sample_shape = tuple(map(int, sample_shape))

    if isinstance(seed, IntLike.__args__):
        rng = jax.random.PRNGKey(seed)
    else:  # key is of type PRNGKey
        rng = seed

    return rng, sample_shape
