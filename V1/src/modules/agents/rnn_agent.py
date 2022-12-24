import torch as th
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic 


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, proxy_z=None):
        if proxy_z is not None:
            assert inputs.shape[:-1] == proxy_z.shape[:-1], ic(inputs.shape, proxy_z.shape)
            inputs = th.cat((inputs, proxy_z), dim=-1)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

