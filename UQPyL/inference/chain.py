import numpy as np

class Chain():
    """
    Fixed-length storage for one inference chain.
    """
    
    def __init__(self, nInput:int, nOutput:int, nCons: int, length:int):
        
        self.length = length
        
        self.nInput = nInput; self.nOutput = nOutput; self.nCons = nCons;
        
        self.decs = np.zeros((length, nInput))
        self.objs = np.zeros((length, nOutput))
        self.logProb = np.zeros(length)
        self.accepted = np.zeros(length, dtype=bool)
        
        self.count = 0
        
        if nCons > 0:
            self.cons = np.zeros((length, nCons))
        else:
            self.cons = None

    def add(self, decs: np.ndarray, objs: np.ndarray, cons: np.ndarray = None,
            logProb: float = None, accepted: bool = True):
        
        self.decs[self.count] = decs
        self.objs[self.count] = objs
        self.logProb[self.count] = np.nan if logProb is None else float(np.ravel(logProb)[0])
        self.accepted[self.count] = bool(accepted)
        
        if self.cons is not None:
            self.cons[self.count] = cons
        
        self.count += 1
