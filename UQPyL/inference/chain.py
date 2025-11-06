import numpy as np

class Chain():
    # Chain
    
    def __init__(self, nInput:int, nOutput:int, nCons: int, length:int):
        
        self.length = length
        
        self.nInput = nInput; self.nOutput = nOutput; self.nCons = nCons;
        
        self.decs = np.zeros((length, nInput))
        self.objs = np.zeros((length, nOutput))
        
        self.count = 0
        
        if nCons > 0:
            self.cons = np.zeros((length, nCons))
        else:
            self.cons = None

    def add(self, decs: np.ndarray, objs: np.ndarray, cons: np.ndarray = None):
        
        self.decs[self.count] = decs
        self.objs[self.count] = objs
        
        if self.cons is not None:
            self.cons[self.count] = cons
        
        self.count += 1
        
    # def burn_in(self, burnIn: int):
              
    #     self.decs = self.decs[burnIn:]
    #     self.objs = self.objs[burnIn:]
        
    #     if self.cons is not None:
    #         self.cons = self.cons[burnIn:]
            
    #     self.count -= burnIn


# class Chains():
    
#     def __init__(self, nChains: int, nInput:int, nOutput:int, nCons: int, length:int):