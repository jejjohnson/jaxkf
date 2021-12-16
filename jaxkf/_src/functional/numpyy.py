# import numpy as np
# from ..distribution import KFDistribution
# from ..state import KFState


# def predictive_mean(mean: np.ndarray, transition: np.ndarray) -> np.ndarray:
#     """predictive mean in update step

#     µ = F µ
#     """

#     mean = transition @ mean
#     return mean


# def predictive_cov(
#     cov: np.ndarray, transition: np.ndarray, noise: np.ndarray
# ) -> np.ndarray:
#     """predictive covariance in update step

#     Σ = F Σ F' + Q
#     """
#     cov = transition @ cov @ transition.T + noise
#     return cov


# def predict_step(state: KFState, dist: KFDistribution) -> KFState:
#     """Prediction step in Kalman filter eqns"""
#     # predictive mean, µ = F µ
#     mean = dist.predictive_mean(state.mean)

#     # predictive covariance, Σ = F Σ F' + Q
#     cov = dist.predictive_cov(state.cov)

#     return mean, cov

# import numpy as np
# from ..state import KFState
# from ..distribution import KFDistribution


# def residual(obs: np.ndarray, state: KFState, dist: KFDistribution) -> np.ndarray:
#     """Error (Residual) between measurement and predictions"""
#     # unroll variables
#     µ = state.mean
#     C = dist.measurement_matrix

#     # predictive mean
#     µ_pred = dist.predictive_mean(µ)

#     # difference
#     obs_pred = C @ µ_pred

#     # residual
#     res = obs - obs_pred

#     return res


# import numpy as np
# from typing import NamedTuple


# class KFState(NamedTuple):
#     mean: np.ndarray
#     cov: np.ndarray
# from dataclasses import dataclass
# from .functional import predictive_cov, predictive_mean
# import numpy as np


# @dataclass
# class KFDistribution:
#     transition_matrix: np.ndarray
#     transition_noise: np.ndarray
#     measurement_matrix: np.ndarray
#     measurement_noise: np.ndarray

#     def predictive_mean(self, x: np.ndarray) -> np.ndarray:
#         return predictive_mean(x, self.measurement_matrix)

#     def predictive_cov(self, cov: np.ndarray) -> np.ndarray:
#         return predictive_cov(cov, self.transition_matrix, self.transition_noise)
