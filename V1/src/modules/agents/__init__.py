REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .mlp_agent import MLPAgent
from .mlp_ns_agent import MLPNSAgent
from .mlp_gen_agent import MLPGenAgent
from .mlp_gen_ns_agent import MLPGenNSAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["mlp_ns"] = MLPNSAgent
REGISTRY["mlp_gen"] = MLPGenAgent
REGISTRY["mlp_gen_ns"] = MLPGenNSAgent
