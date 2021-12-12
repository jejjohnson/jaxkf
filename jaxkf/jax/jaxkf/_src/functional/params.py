import jax.numpy as jnp
from typing import NamedTuple

class KFParams(NamedTuple):
    F : jnp.ndarray
    R : jnp.ndarray
    H : jnp.ndarray
    Q : jnp.ndarray


class State(NamedTuple):
    mu_t: jnp.ndarray
    Sigma_t: jnp.ndarray
    t: int