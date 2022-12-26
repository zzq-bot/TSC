REGISTRY = {}

from .mlp_encoder import MLPEncoder
from .mlp_ns_encoder import MLPNSEncoder
from .gru_encoder import GRUEncoder
from .gru_ns_encoder import GRUNSEncoder
from .lstm_encoder import LSTMEncoder
from .lstm_ns_encoder import LSTMNSEncoder


REGISTRY["mlp"] = MLPEncoder
REGISTRY["mlp_ns"] = MLPNSEncoder
REGISTRY["gru"] = GRUEncoder
REGISTRY["gru_ns"] = GRUNSEncoder
REGISTRY["lstm"] = LSTMEncoder
REGISTRY["lstm_ns"] = LSTMNSEncoder
