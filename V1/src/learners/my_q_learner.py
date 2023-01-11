import copy

import torch as th
import numpy as np
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from icecream import ic
from modules.encoders import REGISTRY as encoder_REGISITRY
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from modules.vi.ns_vi import NSVI
from torch.optim import Adam


class MyQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        agent_param, proxy_encoder_param = mac.parameters()

        self.params = list(agent_param)
        if proxy_encoder_param is not None: # if not use encoder
            self.params += list(proxy_encoder_param)
            
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        #TODO
        if args.use_encoder:
            team_input_dim = self.args.state_shape + (self.args.n_actions) * self.args.n_agents
            self.team_encoder = encoder_REGISITRY[args.team_encoder](args, input_shape=team_input_dim, is_proxy=False)
            self.params += list(self.team_encoder.parameters())
            self.target_team_encoder = copy.deepcopy(self.team_encoder)

            self.vis = NSVI(args)
            self.params += list(self.vis.parameters())
        else:
            self.team_encoder = None
            self.target_team_encoder = None
            self.vis = None

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
        #self.empty_log_tensor = th.Tensor([0]).cuda()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        keys = batch["key"][:, 0].cpu().numpy().squeeze(1) # (batch_size,)

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        team_z_out, mu_out, logvar_out, vi_losses, proxy_z_out = None, None, None, None, None
        # TODO initialize team_encoder_hidden
        if self.args.use_encoder:
            team_z_out = []
            mu_out, logvar_out = [], []
            vi_losses = []
            proxy_z_out = []
        #self.mac.init_hidden(batch.batch_size)
        self.mac.init_hidden(batch.batch_size)
        team_encoder_hidden_states = None
        if ("gru" in self.args.team_encoder or "lstm" in self.args.team_encoder) and self.team_encoder is not None:
            tmp = self.team_encoder.init_hidden()
            if isinstance(tmp, tuple):
                team_encoder_hidden_states = (
                    tmp[0].expand(batch.batch_size, -1),
                    tmp[1].expand(batch.batch_size, -1)
                )
            else:
                team_encoder_hidden_states = tmp.expand(batch.batch_size, -1)

        for t in range(batch.max_seq_length):
            if t == 0:
                lst_ac_onehot = th.zeros_like(batch["actions_onehot"][:, 0])
            else:
                lst_ac_onehot = batch["actions_onehot"][:, t-1]
            agent_outs, proxy_z, mu, logvar = self.mac.forward(batch, t=t)
            #ic(agent_outs.shape) # (batch, n_control, n_actions)
            #ic(batch["state"][:, t].shape)
            #ic(lst_ac_onehot.shape)
            lst_ac_onehot = lst_ac_onehot.view(batch.batch_size, -1)
            if self.team_encoder is not None:
                if team_encoder_hidden_states is not None:
                    team_z, _, _, team_encoder_hidden_states = self.team_encoder.forward(th.cat((batch["state"][:, t], lst_ac_onehot), dim=-1), team_encoder_hidden_states)
                else:
                    team_z, _, _, _ = self.team_encoder.forward(th.cat((batch["state"][:, t], lst_ac_onehot), dim=-1))
            #ic(batch["obs"].shape)
                vi_loss = self.vis.forward(batch["obs"][:, t, :self.args.n_control], batch["actions_onehot"][:, t, :self.args.n_control],
                    team_z, proxy_z, self.mac.encoder_hidden_states)
            else:
                team_z = None
                vi_loss = None
            mac_out.append(agent_outs)
            if self.args.use_encoder:
                mu_out.append(mu) # (bs* n_control, z_dim)
                logvar_out.append(logvar)
                team_z_out.append(team_z)
                vi_losses.append(vi_loss)
                proxy_z_out.append(proxy_z) # (bs, n_control, proxy_z_dim)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions[:, :, :self.args.n_control]).squeeze(3)  # Remove the last dim
        if self.args.use_encoder:
            team_z_out = th.stack(team_z_out[:-1], dim=1)
            proxy_z_out = th.stack(proxy_z_out[:-1], dim=1) # (bs, ep_len-1, n_control, z_dim)
            proxy_z_out_list = [proxy_z_out[:, :, i] for i in range(self.args.n_control)]
            mask_team_z = mask.expand_as(team_z_out)
            team_z_out = team_z_out * mask_team_z # mask (batch_size, ep_len, z_dim)

            mask_proxy_z = mask.expand_as(proxy_z_out_list[0])
            for i in range(self.args.n_control):
                proxy_z_out_list[i] = proxy_z_out_list[i] * mask_proxy_z
            


        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_team_z_out = None
        if self.args.use_encoder:
            target_team_z_out = []
        #self.target_mac.init_hidden(batch.batch_size)
        self.target_mac.init_hidden(batch.batch_size)
        target_team_encoder_hidden_states = None
        if ("gru" in self.args.team_encoder or "lstm" in self.args.team_encoder) and self.team_encoder is not None:
            tmp = self.target_team_encoder.init_hidden()
            if isinstance(tmp, tuple):
                target_team_encoder_hidden_states = (
                    tmp[0].expand(batch.batch_size, -1),
                    tmp[1].expand(batch.batch_size, -1)
                )
            else:
                target_team_encoder_hidden_states = tmp.expand(batch.batch_size, -1)

        for t in range(batch.max_seq_length):
            if t == 0:
                lst_ac_onehot = th.zeros_like(batch["actions_onehot"][:, 0])
            else:
                lst_ac_onehot = batch["actions_onehot"][:, t-1]
            lst_ac_onehot = lst_ac_onehot.view(batch.batch_size, -1)
            target_agent_outs, _, _, _ = self.target_mac.forward(batch, t=t)
            if self.args.use_encoder:
                if target_team_encoder_hidden_states is not None:
                    target_team_z, _, _, target_team_encoder_hidden_states = self.target_team_encoder.forward(th.cat((batch["state"][:, t], lst_ac_onehot), dim=-1), target_team_encoder_hidden_states)
                else:
                    target_team_z, _, _, _ = self.target_team_encoder.forward(th.cat((batch["state"][:, t], lst_ac_onehot), dim=-1))
           
            target_mac_out.append(target_agent_outs)
            if self.args.use_encoder:
                target_team_z_out.append(target_team_z)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        if self.args.use_encoder:
            target_team_z_out = th.stack(target_team_z_out[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:, :self.args.n_control] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions[:, :, :self.args.n_control] == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
    
        ######################################################################
        # 1. Bellman error
        ######################################################################
        # Mix
        if self.mixer is not None:
            if not self.args.use_encoder:
                assert team_z_out is None and target_team_z_out is None
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], team_z_out)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_team_z_out)

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.args.use_encoder:
            ######################################################################
            # 2. Contrastive loss
            ######################################################################
            batch_ep_len = mask.squeeze(-1).sum(-1).unsqueeze(-1)
            team_z_out_mean = team_z_out.mean(dim=1) * batch_ep_len
            self.mac.recorder.update(keys=keys, values=team_z_out_mean)
            for i in range(len(proxy_z_out_list)):
                proxy_z_out_list[i] = proxy_z_out_list[i].mean(dim=1) * batch_ep_len
                self.mac.agents_recorder[i].update(keys=keys, values=proxy_z_out_list[i])

            if self.args.use_contrastive_loss:
                dpp_loss = self.mac.recorder.get_dpp_loss(keys=keys)
                bias_loss = self.mac.recorder.get_bias_loss(keys=keys, values=team_z_out_mean)
                contrastive_loss = self.args.contrastive_lambda_2 * (self.args.contrastive_lambda_1 * dpp_loss + bias_loss) # weighted factor
                agents_dpp_loss = 0
                agents_bias_loss = 0
                agents_contrastive_loss = 0
                if self.args.use_proxy_contrastive_loss:
                    for i in range(len(self.args.n_control)):
                        agents_dpp_loss += self.mac.agents_recorder[i].get_dpp_loss(keys=keys)
                        agents_bias_loss += self.mac.agents_recorder[i].get_bias_loss(keys=keys, values=proxy_z_out_list[i])
                    agents_dpp_loss /= self.args.n_control
                    agents_bias_loss /= self.args.n_control
                    agents_contrastive_loss = self.args.contrastive_lambda_2 * (self.args.contrastive_lambda_1 * agents_dpp_loss + agents_bias_loss)
            else:
                dpp_loss = np.array([0])
                bias_loss = np.array([0])
                contrastive_loss = np.array([0])
                agents_dpp_loss = np.array([0])
                agents_bias_loss = np.array([0])
                agents_contrastive_loss = np.array([0])
            ######################################################################
            # 3、VI loss
            ######################################################################
            if self.args.use_vi:
                mu_out = th.stack(mu_out, dim=1).reshape(-1, self.args.proxy_z_dim)
                logvar_out = th.stack(logvar_out, dim=1).reshape(-1, self.args.proxy_z_dim) # (bs, ep_len, n_control, proxy_z_dim) 
                vi_out = th.stack(vi_losses, dim=1) # (bs, ep_len, n_control,)
                p_ = th.distributions.normal.Normal(mu_out, (0.5 * logvar_out).exp())
                entropy = p_.entropy().clamp_(self.args.min_logvar, self.args.max_logvar).mean()
                vi_loss = self.args.vi_lambda_1 * vi_out.mean() - self.args.vi_lambda_2 * entropy
            else:
                vi_loss = np.array([0])
                entropy = np.array([0])
        else:
            # for logging
            dpp_loss = np.array([0])
            bias_loss = np.array([0])
            contrastive_loss = np.array([0])
            agents_dpp_loss = np.array([0])
            agents_bias_loss = np.array([0])
            agents_contrastive_loss = np.array([0])
            vi_loss = np.array([0])
            entropy = np.array([0])

        # Optimise
        self.optimiser.zero_grad()
        if self.args.use_encoder:
            if self.args.use_contrastive_loss:
                if self.args.use_proxy_contrastive_loss:
                    (td_loss + contrastive_loss + agents_contrastive_loss + vi_loss).backward()
                else:
                    (td_loss + contrastive_loss + vi_loss).backward()
            else:
                (td_loss + vi_loss).backward()
        else:
            td_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat('td_loss', td_loss.item(), t_env)
            self.logger.log_stat('dpp_loss', dpp_loss.item(), t_env)
            self.logger.log_stat('bias_loss', bias_loss.item(), t_env)
            self.logger.log_stat('contrastive_loss', contrastive_loss.item(), t_env)
            self.logger.log_stat('agents_dpp_loss', agents_dpp_loss.item(), t_env)
            self.logger.log_stat('agents_bias_loss', agents_bias_loss.item(), t_env)
            self.logger.log_stat('agents_contrastive_loss', agents_contrastive_loss.item(), t_env)
            self.logger.log_stat('vi_loss', vi_loss.item(), t_env)
            self.logger.log_stat('proxy_encoder_entropy', entropy.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
        #assert 0

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.args.use_encoder:  
            self.target_team_encoder.load_state_dict(self.team_encoder.state_dict())

    def _update_targets_soft(self, tau):
        assert 0, print("sth should be corrected with soft target update")
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.args.use_encoder:
            for target_param, param in zip(self.target_team_encoder.parameters(), self.team_encoder.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.args.use_encoder:
            self.team_encoder.cuda()
            self.target_team_encoder.cuda()
            self.vis.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.args.use_encoder:
            th.save(self.team_encoder.state_dict(), "{}/team_encoder.th".format(path))
            th.save(self.vis.state_dict(), "{}/vis.th".format(path))
            th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.use_encoder:
            self.team_encoder.load_state_dict(th.load("{}/team_encoder.th".format(path), map_location=lambda storage, loc: storage))
            self.vis.load_state_dict(th.load("{}/vis.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
