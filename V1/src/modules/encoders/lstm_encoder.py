import torch as th
import torch.nn.functional as F
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(self, args, input_shape, is_proxy=True) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.args = args
        self.is_proxy = is_proxy
        if self.is_proxy:
            self.output_shape = args.proxy_z_dim
            self.hidden_dim = args.proxy_hidden_dim
        else:
            self.output_shape = args.team_z_dim
            self.hidden_dim = args.team_hidden_dim
        self.fc = nn.Linear(input_shape, self.hidden_dim)
        self.lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.mean = nn.Linear(self.hidden_dim, self.output_shape)
        self.logvar = nn.Linear(self.hidden_dim, self.output_shape)
        
        self.min_logvar = args.min_logvar
        self.max_logvar = args.max_logvar

    def init_hidden(self):
        return (
            self.fc.weight.new(1, self.hidden_dim).zero_(),
            self.fc.weight.new(1, self.hidden_dim).zero_(),
        )

    def forward(self, inputs, h=None):
        assert h is not None and isinstance(h, tuple)
        x = F.leaky_relu(self.fc(inputs))
        hidden_h, hidden_c = self.lstm(self.fc2(x))
        mu, logvar = self.mean(hidden_h), self.logvar(hidden_h)
        logvar = logvar.clamp_(self.min_logvar, self.max_logvar)
        z = self.strategy(mu, logvar)
        return z, mu, logvar, (hidden_h, hidden_c)
    
    def strategy(self, mu, logvar):
        logvar = logvar.clamp_(self.min_logvar, self.max_logvar)
        std = (logvar * 0.5).exp()
        eps = th.randn_like(mu).detach()
        z = mu + eps * std
        return z
    