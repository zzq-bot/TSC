import os

import numpy as np
import torch as th
from icecream import ic

class CRPRecorder:
    def __init__(self, args) -> None:
        self.args = args
        self.xi = args.xi
        self.recorder = {}
        #self.eta = args.eta
        #self.rbf_radius = args.rbf_radius # default 80
        self.kernel = args.kernel
        self.l = 0 # count
        self.M = 0 # num of clusters so far
        self.count_M = [] # each elem should be cur num of cluster m
        self.prototype = []
        self.record_checkpoint_path = []
        self.record_npc_idx = [] # point out npc idx for each elem in each cluster
        self.X_encode = [] # record X used for encoder, we could update cluster center with this X

    def set_module(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def cal_prior(self) -> np.ndarray:
        ret = np.array(self.count_M+[self.xi])
        return ret / np.sum(ret)
    
    def cal_posterior(self, v, X, Y, masks) -> np.ndarray:
        v = th.FloatTensor(v).to(self.args.device)
        with th.no_grad():
            log_prob = self.decoder.cal_log_prob(v, X, Y, masks).detach().cpu().numpy()
        return np.exp(log_prob.item())

    def build_cluster_input(self, cluster_buffer):
        assert cluster_buffer.buffer_size == self.args.num_traj_cluster
        batch = cluster_buffer.sample(cluster_buffer.buffer_size)
        if batch.device != self.args.device:
            batch.to(self.args.device)
        states = batch["state"]
        rewards = batch["reward"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        masks = batch["filled"][:, :-1].float()
        masks[:, 1:] = masks[:, 1:] * (1 - terminated[:, :-1])
        
        X = th.cat([states[:, :-1], actions_onehot.reshape(batch.batch_size, batch.max_seq_length-1, -1)], dim=-1)
        Y = th.cat([rewards, states[:, 1:]], dim=-1)
        return X, Y, masks


    def cluster(self, npc_idx, X, Y, masks, unqiue_token) -> int:
        v_l = self.encoder.forward(X, masks).detach().cpu().numpy() # TODO might take mean on axis 0
        # calculate prior
        prior = self.cal_prior()
        posterior = []
        for v in self.prototype:
            posterior.append(self.cal_posterior(v, X, Y, masks))
        posterior.append(self.cal_posterior(v_l, X, Y, masks))
        prob = prior * posterior

        m = np.argmax(prob)
        if m == self.M: # a new cluster
            self.prototype.append(v_l)
            self.count_M.append(1)
            self.M += 1
            self.record_npc_idx.append([npc_idx])
            self.X_encode.append([(X.detach().cpu(), masks.detach().cpu())])
        else:           # add to old clusters
            self.prototype[m] = (self.prototype[m] * self.count_M[m] + v_l) / (self.count_M[m] + 1)
            self.count_M[m] += 1
            self.record_npc_idx[m].append(npc_idx)
            self.X_encode[m].append((X.detach().cpu(), masks.detach().cpu()))

        save_path = self.build_new_checkpoint(m, unqiue_token) 
        self.l += 1
        return m, save_path

    def update_cluster_center(self):
        if len(self.prototype)==0:
            return
        for i in range(len(self.X_encode)):
            v = np.zeros_like(self.prototype[0])
            for X, masks in self.X_encode[i]:
                X, masks = X.to(self.args.device), masks.to(self.args.device)
                tmp_v = self.encoder.forward(X, masks).detach().cpu().numpy()
                v += tmp_v
            v /= len(self.X_encode[i])
            self.prototype[i] = v
    
    def build_new_checkpoint(self, m, unqiue_token):
        save_path = f"./crp_recorder/{unqiue_token}/cluster_{m}/elem_{self.count_M[m]}"
        os.makedirs(save_path, exist_ok=True)
        if len(self.record_checkpoint_path) <= m:
            self.record_checkpoint_path.append([save_path])
        else:
            self.record_checkpoint_path[m].append(save_path)
        return save_path
    
    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "recorder.npy"), self.recorder)
        np.save(os.path.join(save_path, "count_M.npy"), self.count_M)
        np.save(os.path.join(save_path, "prototype.npy"), self.prototype)
        np.save(os.path.join(save_path, "record_checkpoint_path.npy"), self.record_checkpoint_path)
        np.save(os.path.join(save_path, "record_npc_idx.npy"), self.record_npc_idx)
    
    def load(self, load_path):
        assert os.path.exists(load_path)
        self.recorder = np.load(os.path.join(load_path, "recorder.npy"), allow_pickle=True).item()
        self.count_M = np.load(os.path.join(load_path, "count_M.npy"), allow_pickle=True).tolist()
        self.prototype = np.load(os.path.join(load_path, "prototype.npy"), allow_pickle=True).tolist()
        self.record_checkpoint_path = np.load(os.path.join(load_path, "record_checkpoint_path.npy"), allow_pickle=True).tolist()
        self.record_npc_idx = np.load(os.path.join(load_path, "record_npc_idx.npy"), allow_pickle=True).tolist()
        self.M = len(self.count_M)
        self.l = np.sum(self.count_M)