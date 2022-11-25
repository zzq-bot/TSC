import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_control = args.n_control
        self.state_dim = int(np.prod(args.state_shape))
        self.team_z_dim = args.team_z_dim if self.args.use_encoder else 0
        self.embed_dim = args.mixing_embed_dim

        if not self.args.z_gen_hyper:
            if getattr(args, "hypernet_layers", 1) == 1:
                self.hyper_w_1 = nn.Linear(self.state_dim+self.team_z_dim, self.embed_dim * self.n_control)
                self.hyper_w_final = nn.Linear(self.state_dim+self.team_z_dim, self.embed_dim)
            elif getattr(args, "hypernet_layers", 1) == 2:
                hypernet_embed = self.args.hypernet_embed
                self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim+self.team_z_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim * self.n_control))
                self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim+self.team_z_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim))
            elif getattr(args, "hypernet_layers", 1) > 2:
                raise Exception("Sorry >2 hypernet layers is not implemented!")
            else:
                raise Exception("Error setting number of hypernet layers.")

            # State dependent bias for hidden layer
            self.hyper_b_1 = nn.Linear(self.state_dim+self.team_z_dim, self.embed_dim)

            # V(s) instead of a bias for the last layers
            self.V = nn.Sequential(nn.Linear(self.state_dim+self.team_z_dim, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, 1))
        else:
            assert self.team_z_dim > 0, print("Dont use encoder but assign z_gen_hyper!")
            if getattr(args, "hypernet_layers", 1) == 1:
                self.hyper_w_1 = nn.Linear(self.state_dim+self.team_z_dim, self.embed_dim * self.n_control)
                self.hyper_w_final = nn.Linear(self.team_z_dim, self.embed_dim)
            elif getattr(args, "hypernet_layers", 1) == 2:
                hypernet_embed = self.args.hypernet_embed
                self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim+self.team_z_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim * self.n_control))
                self.hyper_w_final = nn.Sequential(nn.Linear(self.team_z_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim))
            elif getattr(args, "hypernet_layers", 1) > 2:
                raise Exception("Sorry >2 hypernet layers is not implemented!")
            else:
                raise Exception("Error setting number of hypernet layers.")

            # State dependent bias for hidden layer
            self.hyper_b_1 = nn.Linear(self.state_dim+self.team_z_dim, self.embed_dim)

            # V(s) instead of a bias for the last layers
            self.V = nn.Sequential(nn.Linear(self.team_z_dim, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, z=None):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        if z is not None:
            z = z.reshape(-1, self.team_z_dim)
            states = th.cat((states, z), -1)
        agent_qs = agent_qs.view(-1, 1, self.n_control)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_control, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        if self.args.z_gen_hyper:
            assert z is not None
            inputs = z
        else:
            inputs = states
        w_final = th.abs(self.hyper_w_final(inputs))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent biass
        v = self.V(inputs).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot