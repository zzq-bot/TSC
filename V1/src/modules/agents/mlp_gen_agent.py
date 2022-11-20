import torch as th
from torch import nn
import torch.nn.functional as F


class MLPGenAgent(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(args.proxy_z_dim, (args.hidden_dim**2)//2),
            nn.ReLU(),
            nn.Linear((args.hidden_dim**2)//2, args.hidden_dim**2)
        )
        self.hyper_b2 = nn.Linear(args.proxy_z_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
    
    def forward(self, inputs, proxy_z):
        bs = inputs.shape[0]
        assert inputs.shape[:-1] == proxy_z.shape[:-1]
        inputs = th.cat((inputs, proxy_z), dim=-1)
        x = F.relu(self.fc1(inputs)).unsqueeze(1)
        assert tuple(x.shape) == (bs, 1, self.args.hidden_dim)
        w2 = self.hyper_w2(proxy_z).reshape(bs, self.args.hidden_dim, -1)
        b2 = self.hyper_b2(proxy_z).unsqueeze(1)
        x = F.relu(th.bmm(x, w2)+b2)
        q = self.fc3(x).squeeze(1)
        return q