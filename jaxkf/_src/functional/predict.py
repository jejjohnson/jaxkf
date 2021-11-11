import numpy as np
from ..distribution import KFDistribution
from ..state import KFState


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


def predict_step(state: KFState, dist: KFDistribution) -> KFState:
    """Prediction step in Kalman filter eqns"""
    # predictive mean, µ = F µ
    mean = dist.predictive_mean(state.mean)

    # predictive covariance, Σ = F Σ F' + Q
    cov = dist.predictive_cov(state.cov)

    return mean, cov
