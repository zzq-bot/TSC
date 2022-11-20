REGISTRY = {}

from .lbf_heuristic import H1, H2, H3, H4, H5, H6, H7, H8
from .null import NullAgent

REGISTRY["h1"] = H1
REGISTRY["h2"] = H2
REGISTRY["h3"] = H3
REGISTRY["h4"] = H4
REGISTRY["h5"] = H5
REGISTRY["h6"] = H6
REGISTRY["h7"] = H7
REGISTRY["h8"] = H8

REGISTRY["null"] = NullAgent