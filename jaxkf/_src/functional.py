import numpy as np

def predictive_mean(mean: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """predictive mean in update step
    
    µ = F µ
    """
    return transition @ mean