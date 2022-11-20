import torch as th
from torch import nn

from modules.encoders.mlp_encoder import MLPEncoder


class MLPNSEncoder(nn.Module):
    def __init__(self, args, input_shape) -> None:
        # should be proxy only
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_control = args.n_control
        self.input_shape = input_shape
        self.encoders = th.nn.ModuleList([MLPEncoder(args, input_shape, is_proxy=True) for _ in range(self.n_control)])


    def forward(self, inputs, h=None):
        zs, mus, logvars = [], [], []
        if inputs.size(0) == self.n_control:
            for i in range(self.n_control):
                z, mu, logvar, _ = self.encoders[i](inputs[i].unsqueeze(0))
                zs.append(z)
                mus.append(mu)
                logvars.append(logvar)
            return th.cat(zs), th.cat(mus), th.cat(logvars)
        else:
            for i in range(self.n_control):
                inputs = inputs.view(-1, self.n_control, self.input_shape)
                z, mu, logvar, _ = self.encoders[i](inputs[:, i])
                zs.append(z)
                mus.append(mu)
                logvars.append(logvar)
            # [(bs, z_dim), ...]-> (bs, n_control, z_dim) -> (bs*n_control, z_dim)
            return th.cat(zs, dim=-1).view(-1, z.size(-1)), th.cat(mus, dim=-1), th.cat(logvars, dim=-1)
    
    def cuda(self, device=None):
        if not device:
            device = self.args.device
        for e in self.encoders:
            e.cuda(device=device)