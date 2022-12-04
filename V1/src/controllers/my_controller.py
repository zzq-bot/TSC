import copy
import os

import numpy as np
import torch as th
from components.action_selectors import REGISTRY as action_REGISTRY
from components.dynamic_schedule import REGISTRY as dynamic_schedule_REGISTRY
from icecream import ic
from modules.agents import REGISTRY as agent_REGISTRY
from modules.agents.mlp_ns_agent import MLPNSAgent
from modules.encoders import REGISTRY as encoder_REGISTRY
from modules.recorders import REGISTRY as recorder_REGISTRY
from npcs import REGISTRY as npc_REGISTRY
from npcs.mlp_npc import MLPNpc
from npcs.null import NullAgent


# This multi-agent controller shares parameters between agents
class MyMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_control = args.n_control
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self.npc_input_shape = self._get_npc_input_shape(scheme)
        self._build_agents(self.input_shape)
        self._build_encoders(self.input_shape-self.args.proxy_z_dim)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        #self.hidden_states = None
        ic(args.train_schedule)
        ic(args.test_schedule)
        
        self.train_dynamic_schedule = dynamic_schedule_REGISTRY[args.train_schedule](args)
        self.test_dynamic_schedule = dynamic_schedule_REGISTRY[args.test_schedule](args)
        
        self.recorder = recorder_REGISTRY[args.recorder_type](args)
    
    def set_schedule_recorder(self, recorder, mode="train"):
        if mode == 'train':
            self.train_dynamic_schedule.set_recorder(recorder)
        else:
            self.test_dynamic_schedule.set_recorder(recorder)
    

    def build_npc(self, env, test_mode=False):
        #ic("hhhhhhhhhhhh")
        if not test_mode:
            schedule = self.train_dynamic_schedule
        else:
            schedule = self.test_dynamic_schedule
        npc_bool_indices, npc_types, info = schedule.init_build()
        if npc_types != "mlp_ns":
            #ic("build npc, self.npc_bool_indices", npc_bool_indices)
            self.npc = [npc_REGISTRY['null'](self.args.n_actions)] * (self.n_agents-self.n_control)
            self.npc_types = copy.copy(npc_types)
            self.npc_bool_indices = copy.copy(npc_bool_indices) 
            npc_indices = np.argwhere(npc_bool_indices==1).flatten()
            # e.g. [1,0,1] / [1,1,1] / ...
            # correspondingly, npc_types = ['h1', 'h2', 'h1'], ['h1', 'h2']
            assert len(npc_indices) == len(npc_types)
            """ic(npc_indices, npc_types)
            assert 0"""
            for idx, npc_type in zip(npc_indices, npc_types):
                #ic(idx+self.n_control)
                self.npc[idx] = npc_REGISTRY[npc_type](env._env.players[idx+self.n_control])
                """if npc_checkpoints is not None and npc_checkpoints[idx] is not None:
                    self.npc[idx].load(npc_checkpoints[idx])"""
        else:
            # temp name
            cluster_idx, chosen_teammate_checkpoint, chosen_npc_idx = info
            self.npc_types = cluster_idx
            self.npc_bool_indices = npc_bool_indices
            self.npc = MLPNSAgent(input_shape=self.npc_input_shape, args=self.args).to(self.args.device)
            self.npc_mlp_ns_idx = chosen_npc_idx 
            #print(self.npc_mlp_ns_idx)
            if isinstance(self.npc_mlp_ns_idx[0], np.ndarray) or isinstance(self.npc_mlp_ns_idx[0], list):
                self.npc_mlp_ns_idx = self.npc_mlp_ns_idx[0]
                if isinstance(self.npc_mlp_ns_idx[0], np.ndarray) or isinstance(self.npc_mlp_ns_idx[0], list):
                    print("Sth wrong!!")
                    print(self.npc_mlp_ns_idx)
                    print(type(self.npc_mlp_ns_idx))
                    assert 0 
            if (not isinstance(self.npc_mlp_ns_idx, np.ndarray)) and (not isinstance(self.npc_mlp_ns_idx, list)):
                
                print("noooo!!!!!")
                print(self.npc_mlp_ns_idx)
                print(type(self.npc_mlp_ns_idx))
                assert 0
            # different from npc_idx above: 
            # denotes npc idxes in this mlpnsagent module
            self.npc.load_state_dict(th.load(os.path.join(chosen_teammate_checkpoint, "agent.th"),\
                 map_location=lambda storage, loc: storage))                                    
            self.npc.freeze()

    def schedule_npc(self, env, test_mode=False):
        #assert 0
        #ic(self.npc_bool_indices)
        if not test_mode:
            schedule = self.train_dynamic_schedule
        else:
            schedule = self.test_dynamic_schedule
        is_change, npc_bool_indices, npc_types, info = schedule.step()
        if is_change:
            #ic(test_mode)
            #ic(self.npc_bool_indices)
            #ic(npc_bool_indices)
            add_indices = np.argwhere((npc_bool_indices==1) & (self.npc_bool_indices==0)).flatten()
            deleted_indices = np.argwhere((npc_bool_indices==0) & (self.npc_bool_indices==1)).flatten()
            self.npc_bool_indices = copy.copy(npc_bool_indices)
            if npc_type != "mlp_ns":
                npc_indices = np.argwhere(npc_bool_indices==1).flatten()
                self.npc_types = copy.copy(npc_types)
                self.npc = [npc_REGISTRY['null'](self.args.n_actions)] * (self.n_agents-self.n_control)
                #ic(npc_indices, npc_types)
                for idx, npc_type in zip(npc_indices, npc_types):
                    #ic(idx)
                    self.npc[idx] = npc_REGISTRY[npc_type](env._env.players[idx+self.n_control])
                    """if npc_checkpoints is not None and npc_checkpoints[idx] is not None:
                        self.npc[idx].load(npc_checkpoints[idx])"""
            else:
                cluster_idx, chosen_teammate_checkpoint, chosen_npc_idx = info
                self.npc_types = cluster_idx
                self.npc = MLPNSAgent(input_shape=self.npc_input_shape, args=self.args).to(self.args.device)
                self.npc_mlp_ns_idx = chosen_npc_idx 
                if not isinstance(self.npc_mlp_ns_idx[0], int):
                    self.npc_mlp_ns_idx = self.npc_mlp_ns_idx[0]
                if not isinstance(self.npc_mlp_ns_idx[0], int):
                    print("Sth wrong!!")
                    print(self.npc_mlp_ns_idx)
                    assert 0 
                # different from npc_idx above: 
                # denotes npc idxes in this mlpnsagent module
                self.npc.load_state_dict(th.load(os.path.join(chosen_teammate_checkpoint, "agent.th"),\
                    map_location=lambda storage, loc: storage))      
                self.npc.freeze()
                
            return add_indices+self.n_control, deleted_indices+self.n_control

        return [], []

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, env=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs][:, :self.n_control], t_env, test_mode=test_mode)
        npc_actions = self.npc_forward(ep_batch, t_ep, env)
        #TODO, add npc_actions here
        #ic(chosen_actions.shape, npc_actions.shape)
        chosen_actions = th.cat((chosen_actions, npc_actions), dim=1)
        return chosen_actions

    def npc_forward(self, ep_batch, t, env=None):
        chosen_actions = []
        #ic(self.npc)
        #ic(env._env.heuristic_obs)
        if isinstance(self.npc, list):
            for npc_idx, npc in enumerate(self.npc):
                if isinstance(npc, NullAgent):
                    chosen_actions.append(npc.real_step(batch_size=ep_batch.batch_size).to(self.args.device))
                else:
                    assert not isinstance(npc, MLPNpc), print(type(npc))
                    #ic(ep_batch["obs"].shape)
                    #inputs = ep_batch["obs"][:, t].squeeze(0)[npc_idx + self.n_control]
                    chosen_actions.append(npc.real_step(obs=env._env.heuristic_obs[npc_idx+self.n_control]).to(self.args.device))
            #ic(chosen_actions)
        else:
            assert len(self.npc_bool_indices) == self.n_agents - self.n_control, print(len(self.npc_bool_indices))
            for i in range(len(self.npc_bool_indices)):
                if self.npc_bool_indices[i] == False:
                    chosen_actions.append(NullAgent.real_step(batch_size=ep_batch.batch_size).to(self.args.device))
                else:
                    inputs = self._build_npc_inputs(ep_batch, t, i+self.n_control)
                    if len(inputs.shape) == 1:
                        inputs = inputs.unsqueeze(0)
                    idx_of_npc_mac = self.npc_mlp_ns_idx[i]
                    #ic(idx_of_npc_mac, len(self.npc.agents))
                    try:
                        q_val = self.npc.agents[idx_of_npc_mac](inputs)
                    except:
                        print("now npc_mlp_ns_idx:", self.npc_mlp_ns_idx)
                        print("idx of npc_mac:", idx_of_npc_mac)
                        
                        raise Exception(f"sth wrong with idx")
                        
                    action = th.argmax(q_val, dim=-1)
                    #action = self.npc.agents[idx_of_npc_mac](inputs)
                    chosen_actions.append(action)
        
        return th.stack(chosen_actions, dim=1) # (batch_size, n_npc)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.proxy_encoder is not None:
            proxy_z, mu, logvar = self.proxy_encoder(inputs=agent_inputs)
            #ic(agent_inputs.shape, proxy_z.shape)
            #assert 0
            agent_outs = self.agent(agent_inputs, proxy_z)
        else:
            agent_outs = self.agent(agent_inputs, None)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            assert 0
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_control, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        if self.proxy_encoder is not None:
            return agent_outs.view(ep_batch.batch_size, self.n_control, -1), \
                proxy_z.view(ep_batch.batch_size, self.n_control, -1), mu, logvar
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_control, -1), None, None, None
    
    def init_hidden(self, batch_size):
        return
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
    
    def parameters(self):
        if self.proxy_encoder is not None:
            return self.agent.parameters(), self.proxy_encoder.parameters()
        return self.agent.parameters(), None

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        if self.proxy_encoder is not None:
            self.proxy_encoder.load_state_dict(other_mac.proxy_encoder.state_dict())

    def cuda(self):
        self.agent.cuda()
        if self.proxy_encoder is not None:
            self.proxy_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        if self.proxy_encoder is not None:
            th.save(self.proxy_encoder.state_dict(), "{}/proxy_encoder.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.proxy_encoder is not None:
            self.proxy_encoder.load_state_dict(th.load("{}/proxy_encoder.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args, train_teammate=False)

    def _build_encoders(self, input_shape):
        if self.args.use_encoder:
            self.proxy_encoder = encoder_REGISTRY[self.args.proxy_encoder](self.args, input_shape)
        else:
            self.proxy_encoder = None

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t, :self.n_control])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t, :self.n_control]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1, :self.n_control])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_control, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_control, -1) for x in inputs], dim=1)
        return inputs

    def _build_npc_inputs(self, batch, t, npc_idx):
        inputs = []
        inputs.append(batch["obs"][:, t, npc_idx]) # (bs, obs_shape)
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t, npc_idx]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1, npc_idx])
        inputs = th.cat(inputs, dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_control
        if self.args.use_encoder:
            input_shape += self.args.proxy_z_dim

        return input_shape

    def _get_npc_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape