import numpy as np
import torch as th
import torch.nn.functional as F


class TypeRecorder:
    def __init__(self, args) -> None:
        self.str2index = {}
        self.tail_index = 0
        self.recorder = {} # index2hiddenvar
        self.args = args
        self.eta = args.eta
        self.rbf_radius = args.rbf_radius # default 80
        self.kernel = args.kernel
        self.init_val = th.zeros(self.args.team_z_dim).to(self.args.device)
    
    @staticmethod
    def encode(npc_types):
        if isinstance(npc_types, list):
            return '-'.join(npc_types)
        elif isinstance(npc_types, int):
            return "{}".format(npc_types)
        elif isinstance(npc_types, str):
            return npc_types
        else:
            raise Exception("dont support such npc_types: {}".format(npc_types))
            
    def add(self, npc_types):
        str_key = self.encode(npc_types)
        if str_key not in self.str2index:
            self.str2index[str_key] = self.tail_index
            self.recorder[self.tail_index] = th.zeros(self.args.team_z_dim).to(self.args.device)
            self.tail_index +- 1
        return self.str2index[str_key]
    
    def update(self, keys, values):
        key_sort_indice = keys.argsort()
        sorted_keys, sorted_values = keys[key_sort_indice], values[key_sort_indice]
        reduced_key, _, split_indice = np.unique(sorted_keys, return_counts=True, return_index=True)
        splited_values_list = th.split(sorted_values, list(split_indice))
        reduced_values = []
        for x in splited_values_list:
            reduced_values.append(x.mean(dim=0))
        # moving average update
        for key, value in zip(reduced_key, reduced_values):
            assert key in self.recorder
            #print(value.shape)
            if th.equal(self.recorder[key], self.init_val):
                self.recorder[key] = value
            else:
                self.recorder[key] = self.eta * self.recorder[key].detach() + (1-self.eta) * value
    
    def get_values(self, keys=None):
        ret = []
        if keys is None:
            for x in self.recorder.values:
                ret.append(x)
        else:

            for key in keys:
                ret.append(self.recorder[key])
        return th.stack(ret, dim=0)
    
    @staticmethod
    def get_rbf_matrix(data, centers, alpha, element_wise_exp=False):
        """this method comes from
        https://github.com/FanmingL/ESCP/blob/3ab4f44ab2770192f529828dc5face15f12a0f1d/algorithms/RMDM.py"""
        out_shape = th.Size([data.shape[0], centers.shape[0], data.shape[-1]])
        data = data.unsqueeze(1).expand(out_shape)
        centers = centers.unsqueeze(0).expand(out_shape)
        if element_wise_exp:
            mtx = (-(centers - data).pow(2) * alpha).exp().mean(dim=-1, keepdim=False)
        else:
            mtx = (-(centers - data).pow(2) * alpha).sum(dim=-1, keepdim=False).exp()
        return mtx

    def get_dpp_loss(self, keys=None, rbf_radius=None):
        key_sort_indice = keys.argsort()
        sorted_keys = keys[key_sort_indice]
        reduced_key, _, _ = np.unique(sorted_keys, return_counts=True, return_index=True)
        #print(len(keys), len(reduced_key))
        z = self.get_values(reduced_key)
        rbf_radius = self.rbf_radius if rbf_radius is None else rbf_radius
        if self.kernel == 'rbf':
            K = self.get_rbf_matrix(z, z, alpha=rbf_radius, element_wise_exp=False) + th.eye(z.shape[0], device=z.device) * 1e-3
        elif self.kernel == 'rbf_elemenet_wise':
            K = self.get_rbf_matrix(z, z, alpha=rbf_radius, element_wise_exp=True) + th.eye(z.shape[0], device=z.device) * 1e-3
        elif self.kernel == 'inner':
            K = z.matmul(z.t()).exp()
            K = K + th.eye(z.shape[0], device=z.device) * 1e-3
        else:
            assert 0
        loss = -th.logdet(K)
        return loss

    def get_bias_loss(self, keys, values):
        key_sort_indice = keys.argsort()
        sorted_keys, sorted_values = keys[key_sort_indice], values[key_sort_indice]
        reduced_key, _, split_indice = np.unique(sorted_keys, return_counts=True, return_index=True)
        splited_values_list = th.split(sorted_values, list(split_indice))
        reduced_values = []
        for x in splited_values_list:
            #print(x.shape)
            reduced_values.append(x.mean(dim=0))
        reduced_values = th.stack(reduced_values, dim=0)
        with th.no_grad():
            target_values = []
            for key in reduced_key:
                target_values.append(self.recorder[key])
            target_values = th.stack(target_values, dim=0)
        #print(reduced_values.shape, target_values.shape)
        assert reduced_values.shape==target_values.shape, print(reduced_values.shape, target_values.shape)
        loss = F.mse_loss(reduced_values, target_values.detach())
        return loss
    