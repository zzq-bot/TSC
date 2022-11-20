from .q_learner import QLearner
from .my_q_learner import MyQLearner
from .teammate_q_learner import TeammateQLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["my_q_learner"] = MyQLearner
REGISTRY["teammate_q_learner"] = TeammateQLearner