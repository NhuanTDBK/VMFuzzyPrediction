from abc import ABCMeta,abstractmethod
class Experiment():
    @abstractmethod
    def fetch_data(self):
        pass
    @abstractmethod
    def put_queue(self,n_input,dataFeeder):
        pass