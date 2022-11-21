import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.teammate_qmix import TeammateQMixer
from modules.traj_encoder import TransformerEncoder
from modules.decoder.rnn_decoder import RNNDecoder
import torch as th
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd
from icecream import ic

class TeammateQLearner:
    def __init__(self, mac, scheme, logger, args) -> None:
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.teammate_mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.teammate_mixer == "qmix":
                self.mixer = TeammateQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.traj_enc_last_update_T = -1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def set_enc(self, encoder, decoder, enc_params, enc_optimiser):
        self.encoder = encoder
        self.decoder = decoder
        self.enc_params = enc_params
        self.enc_optimiser = enc_optimiser

    def set_recorder(self, recorder):
        self.recorder = recorder

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # The same as q learner
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            if isinstance(agent_outs, tuple):
                agent_outs = agent_outs[0]
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            if isinstance(target_agent_outs, tuple):
                target_agent_outs = target_agent_outs[0]
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        #print(td_error.shape)
        #print(mask.shape)
        mask = mask.expand_as(td_error)
        #print(mask.shape)
    
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()


        # add diversity reg here
        states = batch["state"]
        # use X, Y, masks
        # print(states.shape, actions_onehot.shape, rewards.shape, batch.max_seq_length)
        # assert 0
        X = th.cat([states[:, :-1], actions_onehot.reshape(batch.batch_size, batch.max_seq_length-1, -1)], dim=-1)
        Y = th.cat([rewards, states[:, 1:]], dim=-1)
      
        encoded_z = self.encoder.forward(X, mask)
        v_l = encoded_z.mean(dim=0, keepdim=True)
        diversity_reg = 0
        cur_prototype = self.recorder.prototype
        if len(cur_prototype) > 0:
            # turn -> torch.Tensor
            cur_prototype = th.FloatTensor(np.array(cur_prototype)).to(self.args.device)
            # diversity_reg = - ((cur_prototype-v_l) ** 2).sum() / cur_prototype.size(0)
            diversity_reg = - th.min(((cur_prototype - v_l) ** 2).sum(dim=-1))

        # Optimise
        self.optimiser.zero_grad()
        (td_loss + diversity_reg).backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Update target
        if (episode_num - self.last_target_update_episode) / self.args.teammate_target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("teammate_td_loss", td_loss.item(), t_env)
            diversity_reg = diversity_reg if isinstance(diversity_reg, int) else diversity_reg.item()
            self.logger.log_stat("teammate_diversity_reg", diversity_reg, t_env)
            self.logger.log_stat("teammate_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("teammate_td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("teammate_q_taken_mean",
			                     (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("teammate_target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
			                     t_env)
            self.log_stats_t = t_env
        # Below is the process of optimizing the auto_encoder
        if (t_env - self.traj_enc_last_update_T) / self.args.update_traj_encoder_interval > 1.0:
            """states = batch["states"]
            # use X, Y, masks
            X = th.cat([states[:, :-1], actions], dim=-1)
            Y = th.cat([rewards, states[:, 1:]], dim=-1)
            encoded_z = self.encoder.forward(X, mask)"""
            # Use encoded_z which has been calculated before
            encoded_z = self.encoder.forward(X, mask)
            mle_loss = -self.decoder.cal_log_prob(encoded_z, X, Y, mask)

            self.enc_optimiser.zero_grad()
            mle_loss.backward()
            mle_grad_norm = th.nn.utils.clip_grad_norm_(self.enc_params, self.args.grad_norm_clip)
            self.enc_optimiser.step()
        
            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                self.logger.log_stat("teammate_pred_mle_loss", mle_loss.item(), t_env)
                self.logger.log_stat("teammate_pred_mle_grad_norm", mle_grad_norm, t_env)
    
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")
    
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.encoder.cuda()
        self.decoder.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.encoder.state_dict(), "{}/encoder.th".format(path))
        th.save(self.decoder.state_dict(), "{}/decoder.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
		# Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.encoder.load_state_dict(th.load("{}/encoder.th".format(path), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(th.load("{}/decoder.th".format(path), map_location=lambda storage, loc: storage))

    def load_agent_models(self, path):
        #Only load agent model and mixing network, but keep encoder, decoder unchanged
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    
    def load_enc_models(self, path):
        self.encoder.load_state_dict(th.load("{}/encoder.th".format(path), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(th.load("{}/decoder.th".format(path), map_location=lambda storage, loc: storage))
        