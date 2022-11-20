import argparse

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.token_dim = self._get_token_dim(args)
        # main network structure
        self.transformer = Transformer(self.token_dim, args.enc_emb, args.enc_heads, args.enc_depth, args.enc_emb)
        
    def forward(self, batch_trajectory, trajectory_mask):
        # usually the mask should be "mask" sampled from replaybuffer that indicates length
        outputs, _ = self.transformer.forward(batch_trajectory, trajectory_mask)
        #ic(outputs.shape)
        # mean pooling outputs
        #ic(trajectory_mask.shape)
        repeated_mask = trajectory_mask.expand_as(outputs)
        # 0 if mask, else 1
        #ic(repeated_mask.shape)
        encoded_z = (repeated_mask * outputs).sum(dim=1) / repeated_mask.sum(dim=1)
    
        return encoded_z

    def _get_token_dim(self, args):
        token_dim = args.state_shape
        if args.traj_action:
            token_dim += args.n_actions * args.n_agents
        return token_dim


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask
        assert not mask, "We do not consider mask in this project"

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):
        # x.shape [b, t, e]; mask.shape [b, t]

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = th.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.squeeze(-1)
            # repeat mask
            repeated_mask = mask.unsqueeze(1).repeat(1, h * t, 1)
            repeated_mask = repeated_mask.reshape(b, h, t, t).reshape(b * h, t, t)

            dot = dot.masked_fill(repeated_mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = th.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        attended = self.attention(x, mask)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask


class Transformer(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, mask):

        tokens = self.token_embedding(x)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x, tokens

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = th.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int) # episode_length
    parser.add_argument('--trajectory_attacker_action', default=False, type=bool)
    parser.add_argument('--trajectory_defender_action', default=False, type=bool)
    parser.add_argument('--enc_emb', default=16, type=int)
    parser.add_argument('--enc_heads', default=3, type=int)
    parser.add_argument('--enc_depth', default=2, type=int)
    parser.add_argument('--state_shape', default=10, type=int)
    parser.add_argument('--n_actions', default=4, type=int)
    parser.add_argument('--env_n_actions', default=5, type=int)
    args = parser.parse_args()

    # testing the agent
    encoder = TransformerEncoder(args).cuda()
    batch_size = 3
    state_dim = 10
    ac_dim = 4
    batch = th.rand(batch_size, args.episode, state_dim+ac_dim).cuda()
    batch_mask = th.zeros(batch_size, args.episode).cuda()
    for i in range(batch_size):
        batch[i, -2:] = 0
    z = encoder(batch, batch_mask)