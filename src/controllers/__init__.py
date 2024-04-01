REGISTRY = {}

from .cromac_controller import CromacMAC
from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cromac_mac"] = CromacMAC