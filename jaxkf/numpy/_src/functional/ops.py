import numpy as np
from ..state import KFState
from ..distribution import KFDistribution


def residual(obs: np.ndarray, state: KFState, dist: KFDistribution) -> np.ndarray:
    """Error (Residual) between measurement and predictions"""
    # unroll variables
    µ = state.mean
    C = dist.measurement_matrix

    # predictive mean
    µ_pred = dist.predictive_mean(µ)

    # difference
    obs_pred = C @ µ_pred

    # residual
    res = obs - obs_pred

    return res
