class EstimatedState:
    def __init__(self, memory, current_observation):
        self._m = memory
        self._o = current_observation
    
    @property
    def m(self):
        return self._m

    @property
    def o(self):
        return self._o      

    def get_x(self):
        return tuple(self._m, self._o)
