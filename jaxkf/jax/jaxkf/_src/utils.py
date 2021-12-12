from typing import Tuple
import jax.numpy as jnp


def calculate_error_bounds(
    mu: jnp.ndarray, var: jnp.ndarray, constant: float = 1.96
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate upper and lower bounds for predictive mean and variance."""

    ub = mu + constant * jnp.sqrt(var)
    lb = mu - constant * jnp.sqrt(var)

    return lb, ub
