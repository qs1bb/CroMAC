REGISTRY = {}

from .rnn_agent import RNNAgent
from .cromac_agent import CroMAC

REGISTRY["rnn"] = RNNAgent
REGISTRY["cromac_agent"] = CroMAC