import numpy as np
from icecream import ic

REGISTRY = {}

lbf_fixed_npcs = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]

class BaseSchedule:
    """Fixed npc in the whole training episode"""
    def __init__(self, args) -> None:
        print("use base Schedule")
        if 1:
            self.fixed_npcs = lbf_fixed_npcs
        self.args = args
        self.maximum_npc_num = args.n_agents - args.n_control
        #ic(self.maximum_npc_num)

    def init_build(self):
        self.npc_bool_indices = np.ones(self.maximum_npc_num)
        #ic(self.npc_bool_indices, self.fixed_npcs[:self.maximum_npc_num])
        return self.npc_bool_indices, self.fixed_npcs[:self.maximum_npc_num], None
    
    def step(self):
        #pass, TODO, determine what kinds of info be returned
        return False, self.npc_bool_indices, self.fixed_npcs[:self.maximum_npc_num], None
    
class StaticSchedule:
    """Fixed during one episode"""
    def __init__(self, args) -> None:
        if 1:
            self.fixed_npcs = lbf_fixed_npcs
        self.args = args
        self.maximum_npc_num = args.n_agents - args.n_control
    
    def init_build(self):
        self.npc_bool_indices = np.zeros(self.maximum_npc_num)
        this_epi_npc_num = np.random.choice(range(1, self.maximum_npc_num+1), 1)
        chosen_indices = np.random.choice(range(self.maximum_npc_num), this_epi_npc_num, replace=False)
        self.npc_bool_indices[chosen_indices] = 1

        self.this_epi_npc_types = np.random.choice(a=self.fixed_npcs, size=this_epi_npc_num, replace=True)
        #ic(self.this_epi_npc_types, self.npc_bool_indices)
        #assert 0
        return self.npc_bool_indices, self.this_epi_npc_types, None
    
    def step(self):
        return False, self.npc_bool_indices, self.this_epi_npc_types, None

class FixedDynamicSchedule:
    """Dynamic Schedule (only range from fixed heuristic agents)"""
    def __init__(self, args) -> None:
        if 1:
            self.fixed_npcs = lbf_fixed_npcs
        self.args = args
        self.maximum_npc_num = args.n_agents - args.n_control
        self.active_lower = args.active_lower
        self.active_upper = args.active_upper
        self.waiting_lower = args.waiting_lower
        self.waiting_upper = args.waiting_upper
    
    def init_build(self):
        self.npc_bool_indices = np.zeros(self.maximum_npc_num)
        this_epi_npc_num = np.random.choice(range(1, self.maximum_npc_num+1), 1)[0]
        chosen_indices = np.random.choice(range(self.maximum_npc_num), this_epi_npc_num, replace=False)
        #ic(self.npc_bool_indices, chosen_indices)
       
        self.npc_bool_indices[chosen_indices] = 1

        self.this_epi_npc_types = np.random.choice(a=self.fixed_npcs, size=this_epi_npc_num, replace=True)
        
        self.dict_indice2type = dict(zip(np.argwhere(self.npc_bool_indices==1).flatten(), self.this_epi_npc_types))

        self.active_time = np.random.choice(range(self.active_lower, self.active_upper), self.maximum_npc_num)

        self.waiting_time = np.random.choice(range(self.waiting_lower, self.waiting_upper), 1)[0]

        return self.npc_bool_indices, self.this_epi_npc_types, None
        
    def step(self):
        is_change = False
        # leave
        leave_agent_indices = np.argwhere((self.active_time<=0) & (self.npc_bool_indices==1)).flatten()
        self.active_time -= 1
        add_agent_idx = None
        for idx in leave_agent_indices:
            self.npc_bool_indices[idx] = 0
            is_change = True
            del self.dict_indice2type[idx]
        # add
        can_add = False
        if not np.all(self.npc_bool_indices==1) or len(leave_agent_indices)>0:
            can_add = True
        
        if can_add and self.waiting_time <= 0:
            add_agent_idx = np.random.choice(np.argwhere(self.npc_bool_indices==0).flatten(), 1)[0]
            new_type = np.random.choice(a=self.fixed_npcs, size=1)[0]
            self.dict_indice2type[add_agent_idx] = new_type
            # reset
            self.waiting_time = np.random.choice(range(self.waiting_lower, self.waiting_upper), 1)[0]
        self.waiting_time -= 1
        if add_agent_idx is not None:
            self.npc_bool_indices[add_agent_idx] = 1
            self.active_time[add_agent_idx] = np.random.choice(range(self.active_lower, self.active_upper), 1)[0]
            is_change = True
        self.this_epi_npc_types = np.array(list(self.dict_indice2type.values()))
        return is_change, self.npc_bool_indices, self.this_epi_npc_types, None



class TrainSchedule:
    """Fixed during one episode"""
    def __init__(self, args) -> None:
        self.args = args
        self.maximum_npc_num = args.n_agents - args.n_control
        self.this_epi_npc_types = "mlp_ns"

    def set_recorder(self, recorder):
        self.recorder = recorder
    
    def init_build(self):
        num_clusters = self.recorder.M
        self.this_cluster_idx = np.random.chocie(range(num_clusters), 1)
        num_teammates_of_cluster = self.recorder.count_M[self.this_cluster_idx]
        teammate_idx = np.random.chisquare(range(num_teammates_of_cluster), 1)

        self.this_chosen_teammate_checkpoint = self.recorder.record_checkpoint_path[self.this_cluster_idx][teammate_idx]
        self.this_chosen_npc_idx = self.recorder.record_npc_idx[self.this_cluster_idx][teammate_idx]
        #num_npc = len(chosen_npc_idx)
        
        self.npc_bool_indices = np.zeros(self.maximum_npc_num)
        for i in range(len(self.this_chosen_npc_idx)):
            self.npc_bool_indices[i] = 1
        return self.npc_bool_indices, self.this_epi_npc_types, (self.this_cluster_idx,\
             self.this_chosen_teammate_checkpoint, self.this_chosen_npc_idx)

    def step(self):
        return self.npc_bool_indices, self.this_epi_npc_types, (self.this_cluster_idx,\
             self.this_chosen_teammate_checkpoint, self.this_chosen_npc_idx)


REGISTRY["base"] = BaseSchedule
REGISTRY["static"] = StaticSchedule
REGISTRY["fixed_dynamic"] = FixedDynamicSchedule