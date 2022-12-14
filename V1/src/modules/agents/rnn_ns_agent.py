import torch.nn as nn
from modules.agents.rnn_agent import RNNAgent
import torch as th

from icecream import ic

class RNNNSAgent(nn.Module):
    def __init__(self, input_shape, args, train_teammate=True):
        super(RNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_control = args.n_control
        if train_teammate:
            self.n_control = self.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_control)])

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state, proxy_z=None):
        hiddens = []
        qs = []
        if proxy_z is not None:
            assert inputs.shape[:-1] == proxy_z.shape[:-1], print(inputs.shape, proxy_z.shape)
            if inputs.size(0) == self.n_control: # no ep_batch dim
                for i in range(self.n_control):
                    q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i], proxy_z[i].unsqueeze(0))
                    hiddens.append(h)
                    qs.append(q)
                return th.cat(qs), th.cat(hiddens).unsqueeze(0)
            else:
                for i in range(self.n_control):
                    inputs = inputs.view(-1, self.n_control, self.args.obs_shape)
                    proxy_z = proxy_z.view(-1, self.n_control, self.args.proxy_z_dim)
                    q, h = self.agents[i](inputs[:, i], hidden_state[:, i], proxy_z[:, i])
                    hiddens.append(h.unsqueeze(1))
                    qs.append(q.unsqueeze(1))
                return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)
        else:
            if inputs.size(0) == self.n_control:
                    for i in range(self.n_control):
                        q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                        hiddens.append(h)
                        qs.append(q)
                    return th.cat(qs), th.cat(hiddens).unsqueeze(0)
            else:
                for i in range(self.n_control):
                    inputs = inputs.view(-1, self.n_control, self.input_shape)
                    q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                    hiddens.append(h.unsqueeze(1))
                    qs.append(q.unsqueeze(1))
                return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)
                    
    def cuda(self, device=None):
        if not device:
            device = self.args.device
        for a in self.agents:
            a.cuda(device=device)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = True