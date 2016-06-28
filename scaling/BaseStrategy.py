from abc import abstractmethod
class BaseStrategy(object):
    @abstractmethod
    def allocate_VM(self, resource_used):
        pass
