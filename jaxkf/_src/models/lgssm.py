import treex as tx


class LGSSM(tx.Module):
    transition_matrix: tx.Paramter.node()
    transition_noise: tx.Paramter.node()
    observation_matrix: tx.Parameter.node()
    observation_noise: tx.Parameter.node()

    def __init__(self):
        pass

    def __call__(self):
        pass

    def forward_filter(self):
        """p( z(1:T) | x(1:T-1) )"""
        pass

    def backward_smooth(self):
        """p( z(1:t) | x(1:T) )"""
        pass

    def sample(self):
        """z ~ p()"""
        pass

    def log_prob(self):
        pass

    def posterior_marginals(self):
        pass
