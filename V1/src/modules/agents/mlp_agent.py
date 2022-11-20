import torch as th
from torch import nn
import torch.nn.functional as F
from icecream import ic 


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
        

    def forward(self, inputs, proxy_z=None):
        if proxy_z is not None:
            assert inputs.shape[:-1] == proxy_z.shape[:-1], ic(inputs.shape, proxy_z.shape)
            inputs = th.cat((inputs, proxy_z), dim=-1)
        else:
            # assert self.args.use_encoder is False
            pass
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q
    