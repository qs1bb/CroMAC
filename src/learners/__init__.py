from .cromac_qlearner import CroMAC_QLearner
from .q_learner import QLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["cromac_qlearner"] = CroMAC_QLearner
