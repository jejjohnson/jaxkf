import numpy as np


def predictive_mean(mean: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """predictive mean in update step

    µ = F µ
    """

    mean = transition @ mean
    return mean


def predictive_cov(
    cov: np.ndarray, transition: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    """predictive covariance in update step

    Σ = F Σ F' + Q
    """
    cov = transition @ cov @ transition.T + noise
    return cov
