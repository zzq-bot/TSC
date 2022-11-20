REGISTRY = {}

from .mlp_encoder import MLPEncoder
from .mlp_ns_encoder import MLPNSEncoder
REGISTRY["mlp"] = MLPEncoder
REGISTRY["mlp_ns"] = MLPNSEncoder