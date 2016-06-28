from __future__ import division
import numpy as np
import math
from BaseStrategy import BaseStrategy
class ProactiveSLA(BaseStrategy):
    """Creates scaling decision

    Args:
        max_vms: Maximum number of VMs in the group
        SLA: Service Level Agreement

    Returns:
        Scaling model
    """
    def __init__(self,max_vms = 10,sla = 3, past_consecutive_values = 3, capacity_VM = 0.25,metric=None):
        self.max_vms = max_vms
        self.sla = sla
        self.k = past_consecutive_values
        self.metric = metric
        self.capacity_VM = capacity_VM
    def __allocate_VM(self, res_consump):
        return math.ceil(res_consump / self.capacity_VM)
    def allocate_VMs(self, resource_used):
        self.data_pred = resource_used
        res_alloc = np.zeros(len(resource_used))
        res_alloc = [self.__allocate_VM(item) for item in self.data_pred]
        return res_alloc








