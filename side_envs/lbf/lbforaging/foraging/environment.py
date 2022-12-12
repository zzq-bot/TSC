import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
from icecream import ic

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

        self.active = False

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    
    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"

    def info(self):
        print("my activeness:", self.active)
        print("my position:", self.position)
        print("my score", self.score)
        print("my reward", self.reward)
        print("my level", self.level)
        

class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        #seed=0,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        #self.seed(seed)
        self.players = [Player() for _ in range(players)]
        self.cur_player_num = players
        self.field = np.zeros(field_size, np.int32)

        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]
        # field_size = field_x * field_y

        max_food = self.max_food
        max_food_level = self.max_player_level * len(self.players)
        #TODO, check, modified here
        # n_control + 1
        #max_food_level = self.max_player_level * (2 + 1)

        min_obs = [-1, -1, 0] * max_food + [0, 0, 1] * len(self.players)
        max_obs = [field_x, field_y, max_food_level] * max_food + [
            field_x,
            field_y,
            self.max_player_level,
        ] * len(self.players)

        #return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)
        # we will allow -1 * observation_shape, if the player is not in the env
        return gym.spaces.Box(-1 * np.ones_like(min_obs), np.array(max_obs), dtype=np.float32)
    
    @classmethod
    def from_obs(cls, obs):

        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    @property
    def active_players(self):
        return [p for p in self.players if p.active]

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.active_players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.active_players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_food, max_level):

        food_count = 0
        attempts = 0
        min_level = max_level if self.force_coop else 1

        while food_count < max_food and attempts < 1000:
            attempts += 1
            row = self.np_random.randint(1, self.rows - 1)
            col = self.np_random.randint(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_level
                if min_level == max_level
                else self.np_random.randint(min_level, max_level)
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):

        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    #TODO, check add two methods here
    def add_agent(self, id):
        if self.players[id].active:
            self.logger.warning("Agent already active.")
            assert 0
            return
        attempts = 0
        player = self.players[id]
        while attempts < 1000:
            row = self.np_random.randint(0, self.rows - 1)
            col = self.np_random.randint(0, self.cols - 1)
            if self._is_empty_location(row, col):
                player.setup(
                    (row, col),
                    self.np_random.randint(1, self.max_player_level),
                    self.field_size,
                )
                player.active = True
                break
            attempts += 1
        assert self.players[id].active is True
        self._gen_valid_moves()
        self.cur_player_num += 1

    def remove_agent(self, id):
        self.players[id].active = False
        self.players[id].position = (-1, -1)
        self._gen_valid_moves()
        self.cur_player_num -= 1

    def spawn_players(self, max_player_level):

        for player in self.players:
            if not player.active:
                player.reward = 0
                row = -1
                col = -1
                player.setup(
                    (row, col),
                    -1,
                    self.field_size
                )
                continue
            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows - 1)
                col = self.np_random.randint(0, self.cols - 1)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.randint(1, max_player_level),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        if not player.active:
            return None
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                if a.active and (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                    and (
                        max(
                            self._transform_to_neighborhood(
                                player.position, self.sight, a.position
                            )
                        )
                    )
                    <= 2 * self.sight
                else None for a in self.players
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self, observations):
        #TODO, check. if observation is None....
        def make_obs_array(observation):
            obs = -np.ones(self.observation_space[0].shape, dtype=np.float32)
            if observation is None:
                return obs
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p and p.is_self] + [
                p for p in observation.players if (p and not p.is_self) or (not p)
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 3 * i] = -1
                obs[self.max_food * 3 + 3 * i + 1] = -1
                obs[self.max_food * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                if p:
                    obs[self.max_food * 3 + 3 * i] = p.position[0]
                    obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
                    obs[self.max_food * 3 + 3 * i + 2] = p.level
            #obs[-1] = 1.0
            return obs

        def get_player_reward(observation, idx):
            if observation is None:
                return 0.0
            for p in observation.players:
                if p and p.is_self:
                    return p.reward
            return 0.0
            """if not self.pre_rew_storage is None:
                if observation is None and self.pre_rew_storage[idx] == 0.0:
                    return 0.0
                if observation is None and self.pre_rew_storage[idx] != 0.0:
                    return self.pre_rew_storage[idx]
                for p in observation.players:
                    if p and p.is_self:
                        return p.reward
            else:
                if observation is None:
                    return 0.0
                for p in observation.players:
                    if p and p.is_self:
                        return p.reward"""

        nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs, idx) for idx, obs in enumerate(observations)]
        ndone = [obs.game_over if obs is not None else False for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}

        return nobs, nreward, ndone, ninfo

    def reset(self, active_agents_id):
        #print("hhhhhhhhhhhhh")
        #ic(active_agents_id)
        if active_agents_id is None:
            active_agents_id = list(range(self.n_agents))
        #ic(active_agents_id)
        for player in self.players:
            player.active = False

        for id in active_agents_id:
            self.players[id].active = True

        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        
        player_levels = sorted([player.level for player in self.players if player.active])

        self.spawn_food(
            self.max_food, max_level=sum(player_levels[:3])
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        self.pre_rew_storage = None

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        self.heuristic_obs = observations
        self.is_heuristic_obs_None = [obs is None for obs in self.heuristic_obs]
        for idx in range(len(self.heuristic_obs)):
            if self.heuristic_obs is None:
                if self.players[idx].active is False:
                    assert 0, ic("sth wrong with observation/add/remove")
        
        return nobs

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        #ic(actions)
        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions) if p.active # p should be alive, else do nothing
        ]
        #ic(actions)
        assert len(actions) == len(self.active_players)

        # check if actions are valid
        """for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE"""

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.active_players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.active_players:
            p.score += p.reward

        #TODO, check, let observation = None if player is not active.
        self.pre_rew_storage = [player.reward for player in self.players]
        observations = [self._make_obs(player) for player in self.players]
        #nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        self.heuristic_obs = observations
        self.is_heuristic_obs_None = [obs is None for obs in self.heuristic_obs]
        for idx in range(len(self.heuristic_obs)):
            if self.heuristic_obs is None:
                if self.players[idx].active is False:
                    assert 0, ic("sth wrong with observation/add/remove")

        return self._make_gym_obs(observations)
    
    def update_obs_after_schedule(self):
        observations = [self._make_obs(player) for player in self.players]
        #nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        self.heuristic_obs = observations
        self.is_heuristic_obs_None = [obs is None for obs in self.heuristic_obs]
        for idx in range(len(self.heuristic_obs)):
            if self.heuristic_obs is None:
                if self.players[idx].active is False:
                    assert 0, ic("sth wrong with observation/add/remove")

    def _init_render(self):
        from .rendering import Viewer
        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
