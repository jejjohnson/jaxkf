from dataclasses import dataclass
from .functional import predictive_cov, predictive_mean
import numpy as np


@dataclass
class KFDistribution:
    transition_matrix: np.ndarray
    transition_noise: np.ndarray
    measurement_matrix: np.ndarray
    measurement_noise: np.ndarray

    def predictive_mean(self, x: np.ndarray) -> np.ndarray:
        return predictive_mean(x, self.measurement_matrix)

    def predictive_cov(self, cov: np.ndarray) -> np.ndarray:
        return predictive_cov(cov, self.transition_matrix, self.transition_noise)
