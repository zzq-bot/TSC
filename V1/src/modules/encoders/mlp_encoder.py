import torch as th
import torch.nn.functional as F
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, args, input_shape, is_proxy=True) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.args = args
        self.is_proxy = is_proxy
        if self.is_proxy:
            self.output_shape = args.proxy_z_dim
            hidden_dim = args.proxy_hidden_dim
        else:
            self.output_shape = args.team_z_dim
            hidden_dim = args.team_hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, self.output_shape)
        self.logvar = nn.Linear(hidden_dim, self.output_shape)
        
        self.min_logvar = args.min_logvar
        self.max_logvar = args.max_logvar

    def forward(self, inputs, h=None):
        x = F.leaky_relu(self.fc1(inputs))
        x = F.leaky_relu(self.fc2(x))
        mu, logvar = self.mean(x), self.logvar(x)
        logvar = logvar.clamp_(self.min_logvar, self.max_logvar)
        z = self.strategy(mu, logvar)
        return z, mu, logvar, None
    
    def strategy(self, mu, logvar):
        logvar = logvar.clamp_(self.min_logvar, self.max_logvar)
        std = (logvar * 0.5).exp()
        eps = th.randn_like(mu).detach()
        z = mu + eps * std
        return z
    
    """def ep_batch_forward(self, batch, t):
        inputs = batch["state"][:, t]
        return self.forward(inputs)"""
    