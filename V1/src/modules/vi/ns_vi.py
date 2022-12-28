import torch as th
from icecream import ic
from modules.vi.vi import VI
from torch import nn


class NSVI(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.n_control = args.n_control
        self.vis = th.nn.ModuleList([VI(args) for _ in range(self.n_control)])
    
    def forward(self, obs, ac, team_z, proxy_z, hiddens=None):
        if hiddens is not None and isinstance(hiddens, tuple):
            hiddens = hiddens[0] # we neglect cell state if using lstm encoder
        n_log_probs = []
        if obs.size(0) == self.n_control:
            for i in range(self.n_control):
                # just in case
                if len(hiddens.shape) == 3:
                    assert hiddens.size(0) == 1 and hiddens.size(1) == self.n_control
                    hiddens = hiddens[0]
                else:
                    assert len(hiddens.shape) == 2
                    assert hiddens.size(0) == self.n_control

                n_log_prob = self.vis[i](obs[i].unsqueeze(0), ac[i].unsqueeze(0),\
                    team_z, proxy_z.unsqueeze(0), hiddens[i].unsqueeze(0))
                n_log_probs.append(n_log_prob)
            return th.cat(n_log_probs)
        else:
            for i in range(self.n_control):
                obs = obs.view(-1, self.n_control, self.args.obs_shape)
                ac = ac.view(-1, self.n_control, self.args.n_actions)
                team_z = team_z.view(-1, self.args.team_z_dim)
                proxy_z = proxy_z.view(-1, self.n_control, self.args.proxy_z_dim)
                hiddens = hiddens.view(-1, self.n_control, self.args.proxy_hidden_dim)
                #ic(obs.shape, ac.shape, team_z.shape, proxy_z.shape)
                n_log_prob = self.vis[i](obs[:, i], ac[:, i], team_z, proxy_z[:, i], hiddens[:, i])
                n_log_probs.append(n_log_prob)
            return th.cat(n_log_probs, dim=-1).view(-1, n_log_prob.size(-1))

    def cuda(self, device=None):
        if not device:
            device = self.args.device
        for vi in self.vis:
            vi.cuda(device=device)