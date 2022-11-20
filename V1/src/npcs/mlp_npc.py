from torch import nn
import torch.nn.functional as F

class MLPNpc(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
    
    def forward(self, inputs):
        #assert inputs.shape[:-1] == proxy_z.shape[:-1]
        #inputs = th.cat((inputs, proxy_z), dim=-1)
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q
        