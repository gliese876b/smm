class Event:
    def __init__(self, observation, action):
        self._o = tuple(observation)
        self._a = action
        
    @property
    def o(self):
        return self._o        

    @property
    def a(self):
        return self._a  
        
    def get_e(self):
        if self._a:
            return tuple(self._o, self._a)
        return self._o