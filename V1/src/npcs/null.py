import torch


class NullAgent:
    def __init__(self, n_actions) -> None:
        self.n_actions = n_actions

    def step(self, batch_size):
        return torch.zeros(batch_size, )
        
    def real_step(self, batch_size):
        #assert obs.shape[0] == 1, print(obs.shape)
        return torch.zeros(batch_size, )# * self.n_actions