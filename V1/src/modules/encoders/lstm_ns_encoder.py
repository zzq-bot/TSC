import torch as th
from torch import nn

from modules.encoders.lstm_encoder import LSTMEncoder


class LSTMNSEncoder(nn.Module):
    def __init__(self, args, input_shape) -> None:
        # should be proxy only
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_control = args.n_control
        self.input_shape = input_shape
        self.encoders = th.nn.ModuleList([LSTMEncoder(args, input_shape, is_proxy=True) for _ in range(self.n_control)])

    def init_hidden(self):
        tmp = [e.init_hidden() for e in self.encoders]
        h = [hidden[0] for hidden in tmp]
        c = [hidden[1] for hidden in tmp]
        return h, c

    def forward(self, inputs, h=None):
        assert h is not None and isinstance(h, tuple)
        h, c = h
        zs, mus, logvars = [], [], []
        ret_h, ret_c = [], []
        if inputs.size(0) == self.n_control:
            for i in range(self.n_control):
                z, mu, logvar, hidden = self.encoders[i](inputs[i].unsqueeze(0), (h[i].unsqueeze(0), c[i].unsqueeze(0)))
                zs.append(z)
                mus.append(mu)
                logvars.append(logvar)
                ret_h.append(hidden[0])
                ret_c.append(hidden[1])
            return th.cat(zs), th.cat(mus), th.cat(logvars), (th.cat(ret_h), th.cat(ret_c))
        else:
            for i in range(self.n_control):
                inputs = inputs.view(-1, self.n_control, self.input_shape)
                z, mu, logvar, hidden = self.encoders[i](inputs[:, i], (h[:, i], c[:, i]))
                zs.append(z)
                mus.append(mu)
                logvars.append(logvar)
                ret_h.append(hidden[0])
                ret_c.append(hidden[1])
            # [(bs, z_dim), ...]-> (bs, n_control, z_dim) -> (bs*n_control, z_dim)
            return th.cat(zs, dim=-1).view(-1, z.size(-1)), th.cat(mus, dim=-1), th.cat(logvars, dim=-1), \
                (th.cat(ret_h, dim=-1), th.cat(ret_c, dim=-1))
    
    def cuda(self, device=None):
        if not device:
            device = self.args.device
        for e in self.encoders:
            e.cuda(device=device)