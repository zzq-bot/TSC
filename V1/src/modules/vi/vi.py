import torch as th
import torch.nn.functional as F
from icecream import ic
from torch import nn


class VI(nn.Module):
    #I(proxy_z^i;tse_z|o_t^i, a_{t-1}^i, (maybe also hidden))
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.input_dim = args.obs_shape + args.n_actions + args.team_z_dim
        if "gru" in self.args.team_encoder or "lstm" in self.args.team_encoder:
            self.input_dim += args.proxy_hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.args.vi_hidden_dim)
        self.fc2 = nn.Linear(self.args.vi_hidden_dim, self.args.vi_hidden_dim)
        self.mean = nn.Linear(self.args.vi_hidden_dim, self.args.proxy_z_dim)
        self.logvar = nn.Linear(self.args.vi_hidden_dim, self.args.proxy_z_dim)
    
    def forward(self, obs, ac, team_z, proxy_z, hidden=None):
        if hidden is None:
            vi_input = th.cat((obs, ac, team_z), dim=-1)
        else:
            vi_input = th.cat((obs, ac, team_z, hidden), dim=-1)
        h = self.fc2(F.leaky_relu(self.fc1(vi_input)))
        mu, logvar = self.mean(h), self.logvar(h)
        logvar = logvar.clamp_(self.args.min_logvar, self.args.max_logvar)
        dist = th.distributions.normal.Normal(mu, (0.5 * logvar).exp())
        log_prob = dist.log_prob(proxy_z).clamp_(-1000, 0)
        log_prob = log_prob.sum(-1)
        #assert 0
        return -log_prob