import torch as th
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic


class RNNDecoder(nn.Module):
    def __init__(self, args):
        super(RNNDecoder, self).__init__()
        self.args = args
        input_dim, output_dim = self._get_shapes(args)
        self.fc1 = nn.Linear(input_dim, args.dec_emb)
        self.rnn = nn.GRUCell(args.dec_emb, args.dec_emb)
        self.fc2 = nn.Linear(args.dec_emb + args.enc_emb, args.dec_emb)

        self.output_mean = nn.Linear(args.dec_emb, output_dim)
        self.output_logvar = nn.Linear(args.dec_emb, output_dim)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.dec_emb).zero_()

    def forward(self, encoded_z, inputs, hidden_state):
        """
        encoded_z.shape: [bs, z_dim]
        state_input.shape: [bs, state_dim]
        hidden_state.shape: [bs, h_dim]
        """
        x = F.relu(self.fc1(inputs))
        hidden_state = self.rnn(x, hidden_state)

        x = th.cat([hidden_state, encoded_z], dim=1)
        x = F.relu(self.fc2(x))
        
        mu, logvar = self.output_mean(x), self.output_logvar(x)
        logvar = logvar.clamp_(self.args.min_logvar, self.args.max_logvar)
        #dist = th.distributions.normal.Normal(mu, (0.5 * logvar).exp())
        decoded_output = self.strategy(mu, logvar)
        #decoded_output = self.output_layer(x)
        return decoded_output, hidden_state, mu, logvar
    
    def strategy(self, mu, logvar):
        logvar = logvar.clamp_(self.args.min_logvar, self.args.max_logvar)
        std = (logvar * 0.5).exp()
        eps = th.randn_like(mu).detach()
        z = mu + eps * std
        return z

    def cal_log_prob(self, encoded_z, traj_inputs, targeted_traj_outputs, masks):
        log_probs = []
        bs, T, _ = traj_inputs.size()
        hidden_state = self.init_hidden().repeat(bs, 1)
        
        for t in range(T-1):
            inputs = traj_inputs[:, t]
            targets = targeted_traj_outputs[:, t+1]
            #ic(targets.shape)
            _, hidden_state, mu, logvar = self.forward(encoded_z, inputs, hidden_state)
            
            dist = th.distributions.normal.Normal(mu, (0.5 * logvar).exp())
            log_prob = dist.log_prob(targets).clamp_(-1000, 0).mean(dim=-1, keepdim=True) # (batch_size, 1)
            #ic(log_prob.shape)
            #assert 0
            log_probs.append(log_prob)
        log_probs = th.stack(log_probs, dim=1) # (batch_size, T-1, 1)  
        if masks.shape[1] == log_probs.shape[1] + 1:
            masks = masks[:, :-1]
        # Just check. 
        # TODO delete
        elif masks.shape[1] == log_probs.shape[1]:
            pass
        else:
            assert 0
        repeated_mask = masks.expand_as(log_probs)
        ret_log_prob = (log_probs * repeated_mask).sum() / repeated_mask.sum()
        return ret_log_prob

    def _get_shapes(self, args):
        """
        return the input_shape and output_shape
        """
        input_dim = args.state_shape
        if args.traj_action:
            input_dim += args.n_actions * args.n_agents
        output_dim = args.state_shape
        if args.traj_reward:
            output_dim += 1
        return input_dim, output_dim
    