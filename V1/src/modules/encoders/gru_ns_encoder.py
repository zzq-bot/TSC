import torch as th
from torch import nn

from modules.encoders.gru_encoder import GRUEncoder


class GRUNSEncoder(nn.Module):
    def __init__(self, args, input_shape) -> None:
        # should be proxy only
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_control = args.n_control
        self.input_shape = input_shape
        self.encoders = th.nn.ModuleList([GRUEncoder(args, input_shape, is_proxy=True) for _ in range(self.n_control)])


    def init_hidden(self):
        return th.cat([e.init_hidden() for e in self.encoders])

    def forward(self, inputs, h):
        assert isinstance(h, tuple)
        zs, mus, logvars = [], [], []
        ret_h = []
        if inputs.size(0) == self.n_control:
            for i in range(self.n_control):
                z, mu, logvar, hidden = self.encoders[i](inputs[i].unsqueeze(0), h[i].unsqueeze(0))
                zs.append(z)
                mus.append(mu)
                logvars.append(logvar)
                ret_h.append(hidden)
            return th.cat(zs), th.cat(mus), th.cat(logvars), th.cat(ret_h)
        else:
            for i in range(self.n_control):
                inputs = inputs.view(-1, self.n_control, self.input_shape)
                z, mu, logvar, hidden = self.encoders[i](inputs[:, i], h[:, i])
                zs.append(z.unsqueeze(1))
                mus.append(mu.unsqueeze(1))
                logvars.append(logvar.unsqueeze(1))
                ret_h.append(hidden.unsqueeze(1))
            return th.cat(zs, dim=-1).view(-1, z.size(-1)),\
                th.cat(mus, dim=-1).view(-1, mu.size(-1)),\
                th.cat(logvars, dim=-1).view(-1, logvar.size(-1)),\
                th.cat(ret_h, dim=1)