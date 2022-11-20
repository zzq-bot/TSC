import torch.nn as nn
from modules.agents.mlp_gen_agent import MLPGenAgent
import torch as th

class MLPGenNSAgent(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_control = args.n_control
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([MLPGenAgent(input_shape, args) for _ in range(self.n_control)])

    def forward(self, inputs, proxy_z):
        assert inputs.shape[:-1] == proxy_z.shape[:-1]
        qs = []
        if inputs.size(0) == self.n_control:
            for i in range(self.n_control):
                q = self.agents[i](inputs[i].unsqueeze(0), proxy_z[:, i])
                qs.append(q)
            return th.cat(qs)
        else:
            for i in range(self.n_control):
                inputs = inputs.view(-1, self.n_control, self.input_shape)
                q = self.agents[i](inputs[:, i], proxy_z[:, i])
                qs.append(q.unsqueeze(1))
            return th.cat(qs, dim=-1).view(-1, q.size(-1))

    def cuda(self, device=None):
        if not device:
            device = self.args.device
        for a in self.agents:
            a.cuda(device=device)