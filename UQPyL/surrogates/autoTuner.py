from typing import Literal


from ..optimization.algorithmABC import Algorithm
from .surrogateABC import Surrogate

class autoTuner():
    def __init__(self, optimizer: Algorithm, model: Surrogate):
        
        self.optimizer = optimizer

        self.model = model
        
    def tune(self, paraList):
        pass
    
    def getParaList(self):
        
        return list(self.model.setting.parasValue.keys())