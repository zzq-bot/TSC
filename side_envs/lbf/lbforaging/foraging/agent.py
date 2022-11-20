import logging

import numpy as np

_MAX_INT = 999999


class Agent:
    name = "Prototype Agent"

    def __repr__(self):
        return self.name

    def __init__(self, player):
        self.logger = logging.getLogger(__name__)
        self.player = player
        #print(self.player.active)
        #assert 0

    def __getattr__(self, item):
        return getattr(self.player, item)

    def _step(self, obs):
        self.observed_position = next(
            (x for x in obs.players if x and x.is_self), None
        ).position

        # saves the action to the history
        action = self.step(obs)
        # self.history.append(action)

        return action

    def step(self, obs):
        raise NotImplemented("You must implement an agent")

    def _closest_food(self, obs, max_food_level=None, start=None):
        #print(self.player.active)
        if start is None:
            x, y = self.observed_position
        else:
            x, y = start

        field = np.copy(obs.field)

        if max_food_level:
            field[field > max_food_level] = 0

        r, c = np.nonzero(field)
        try:
            min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
        except ValueError:
            return None

        return r[min_idx], c[min_idx]

    def _farthest_food(self, obs, max_food_level=None, start=None):

        if start is None:
            x, y = self.observed_position
        else:
            x, y = start

        field = np.copy(obs.field)

        if max_food_level:
            field[field > max_food_level] = 0

        r, c = np.nonzero(field)
        try:
            max_idx = ((r - x) ** 2 + (c - y) ** 2).argmax()
        except ValueError:
            return None

        return r[max_idx], c[max_idx]

    def _highest_eligible_food(self, obs, max_food_level):
        field = np.copy(obs.field)

        if ((field <= max_food_level) & (field != 0)).any():
            field[field > max_food_level] = 0

        r, c = np.nonzero(field)
        if not len(r) == 0:
            max_r, max_c = -1, -1
            max_level = float('-inf')
            for r_v, c_v in zip(r,c):
                if max_level < field[r_v][c_v]:
                    max_level = field[r_v][c_v]
                    max_r, max_c = r_v, c_v

            return max_r, max_c

        return None

    def _make_state(self, obs):

        state = str(obs.field)
        for c in ["]", "[", " ", "\n"]:
            state = state.replace(c, "")

        for a in obs.players:
            state = state + str(a.position[0]) + str(a.position[1]) + str(a.level)

        return int(state)

    def cleanup(self):
        pass
