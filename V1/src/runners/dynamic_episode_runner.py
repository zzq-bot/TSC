from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from icecream import ic

class DynamicEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        ic(self.args.env)
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        self.prefix = ""

    def set_prefix(self, prefix):
        self.prefix = prefix

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, test_mode=False):
        self.batch = self.new_batch()
        self.mac.build_npc(self.env, test_mode)
        active_agents_id = np.argwhere(self.mac.npc_bool_indices==1).flatten() + self.args.n_control
        #ic(active_agents_id)
        self.env.reset(list(range(self.args.n_control)) + active_agents_id.tolist())
        self.t = 0
        #if not test_mode:
        

    def run(self, test_mode=False):
        self.reset(test_mode)
        
        terminated = False
        episode_return = 0
        #self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            #ic(self.t)
            if isinstance(self.mac.npc_types, list):
                key = self.mac.recorder.add(sorted(self.mac.npc_types))
            else:
                key = self.mac.recorder.add(self.mac.npc_types)
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "key": [(key, )]
            }
            #ic(key)
            #ic(self.mac.npc)
            #ic(self.env._env.is_heuristic_obs_None)
            self.batch.update(pre_transition_data, ts=self.t)
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, env=self.env)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            # TODO, dynamic schedule for env
            # check
            add_indices, deleted_indices = self.mac.schedule_npc(self.env, test_mode)
            #ic(add_indices)
            #ic(self.env._env.is_heuristic_obs_None)
            #ic(deleted_indices)
            #if len(add_indices)>0:
            for id in add_indices:
                self.env.add_agent(id)
            #if len(deleted_indices)>0:
            for id in deleted_indices:
                self.env.remove_agent(id)
            if len(add_indices) > 0 or len(deleted_indices) > 0:
                self.env._env.update_obs_after_schedule()

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, env=self.env)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = f"{self.prefix}_test_" if test_mode else f"{self.prefix}"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
