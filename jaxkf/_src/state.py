import numpy as np
from typing import NamedTuple


class KFState(NamedTuple):
    mean: np.ndarray
    cov: np.ndarray
