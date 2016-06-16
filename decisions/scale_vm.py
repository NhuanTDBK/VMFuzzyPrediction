import numpy as np
import math

class Scaling(object):
    """Creates scaling decision

    Args:
        max_vms: Maximum number of VMs in the group
        SLA: Service Level Agreement

    Returns:
        Scaling model
    """
    def __init__(self,max_vms = 10,sla = 3, past_consecutive_values = 3, capacity_VM = 0.25):
        self.max_vms = max_vms
        self.sla = sla
        self.k = past_consecutive_values
    def allocate(self,cpu_util=None,last_cpu_utils=None):
        self.cpu_util = cpu_util



