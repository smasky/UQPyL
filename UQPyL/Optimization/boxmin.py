import numpy as np
from typing import Callable

class Boxmin():
    type='MP' #mathematical programming
    def __init__(self, initialPos: np.ndarray, ub: np.ndarray, lb: np.ndarray) -> None:
        self.initalPos=initialPos
        self.ub=ub
        self.lb=lb
        self.nv=0
    ###################################Interface Function#################################
    def run(self, func: Callable):
        self.func=func
        
        self._start()
        p=self.initalPos.size
        
        for _ in range(np.minimum(p,4)):
            pos_copy=self.pos.copy()
            self._explore()
            self._move(pos_copy)
            
        return self.pos, self.f
    #######################################Private Function################################
    def _move(self, pos_old: np.ndarray):
        
        pos=self.pos.copy()
        f=self.f
        
        v=pos/pos_old
        
        rept=True
        while rept:
            pos_c=np.minimum(self.ub, np.maximum(self.lb, pos*v))
            ff=self.func(pos_c)
            self.nv+=1
            
            if ff<f:
                pos=pos_c.copy()
                f=ff
                v=v**2
            else:
                rept=False
                
        self.D = self.D[np.r_[1:pos.size, 0]] ** 0.25
        self.f=f
        self.pos=pos.copy()
            
    def _explore(self):
        
        pos=self.pos.copy()
        f=self.f
        
        for k in np.arange(0,pos.size):
            pos_c=pos.copy()
            DD=self.D[k]
            
            if pos[k]==self.ub[k]:
                atbd=True
                pos_c[k]=pos[k]/np.sqrt(DD)
            elif pos[k]==self.lb[k]:
                atbd=True
                pos_c[k]=pos[k]*np.sqrt(DD)
            else:
                atbd=False
                pos_c[k]=np.minimum(self.ub[k], pos[k]*DD)
            
            ff=self.func(pos_c)
            self.nv+=1
            
            if ff<f:
                pos=pos_c.copy()
                f=ff
            else:
                if not atbd:
                    pos_c[k] = np.maximum(self.lb[k], pos[k] / DD)
                    ff = self.func(pos_c)
                    self.nv+=1
                    if ff<f:
                        pos=pos_c.copy()
                        f=ff
        self.pos=pos
        self.f=f
        
    def _start(self):
        
        p = self.initalPos.size
        D = 2 ** (np.arange(1, p + 1).reshape(-1, 1) / (p + 2))
        
        f = self.func(self.initalPos)
        
        self.nv=1
        self.D=D
        self.f=f
        self.pos=self.initalPos