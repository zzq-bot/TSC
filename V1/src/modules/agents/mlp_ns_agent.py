import torch.nn as nn
from modules.agents.mlp_agent import MLPAgent
import torch as th
from icecream import ic

class MLPNSAgent(nn.Module):
    def __init__(self, input_shape, args, train_teammate=True) -> None:
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_control = args.n_control
        if train_teammate:
            self.n_control = self.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([MLPAgent(input_shape, args) for _ in range(self.n_control)])
        ic("length of MLPNS Agent:", len(self.agents))

    def forward(self, inputs, proxy_z=None):
        qs = []
        if proxy_z is not None:
            assert inputs.shape[:-1] == proxy_z.shape[:-1]
            if inputs.size(0) == self.n_control:
                for i in range(self.n_control):
                    #ic(self.input_shape)
                    #ic(inputs[i].unsqueeze(0).shape, proxy_z[i].unsqueeze(0).shape)
                    q = self.agents[i](inputs[i].unsqueeze(0), proxy_z[i].unsqueeze(0))
                    #assert 0
                    qs.append(q)
                return th.cat(qs)
            else:
                for i in range(self.n_control):
                    inputs = inputs.view(-1, self.n_control, self.args.obs_shape)
                    proxy_z = proxy_z.view(-1, self.n_control, self.args.proxy_z_dim)
                    q = self.agents[i](inputs[:, i], proxy_z[:, i])
                    qs.append(q.unsqueeze(1))
                return th.cat(qs, dim=-1).view(-1, q.size(-1))
        else:
            #assert self.args.use_encoder is False
            if inputs.size(0) == self.n_control:
                for i in range(self.n_control):
                    #ic(self.input_shape)
                    #ic(inputs[i].unsqueeze(0).shape, proxy_z[i].unsqueeze(0).shape)
                    q = self.agents[i](inputs[i].unsqueeze(0), None)
                    #assert 0
                    qs.append(q)
                return th.cat(qs)
            else:
                for i in range(self.n_control):
                    inputs = inputs.view(-1, self.n_control, self.args.obs_shape)
                    q = self.agents[i](inputs[:, i], None)
                    qs.append(q.unsqueeze(1))
                return th.cat(qs, dim=-1).view(-1, q.size(-1))

    def cuda(self, device=None):
        if not device:
            device = self.args.device
        for a in self.agents:
            a.cuda(device=device)