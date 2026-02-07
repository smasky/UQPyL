import numpy as np
class Boxmin():
    
    type = "MP"
    
    def __init__(self) -> None:
        
        pass
        
    ###################################Interface Function#################################
    def run(self, problem, xInit=None):
        
        self.ub=problem.ub.ravel()
        self.lb=problem.lb.ravel()
        
        if xInit is None:
            xInit=np.random.uniform(problem.lb.ravel(), problem.ub.ravel(), problem.nInput)
            
        self.func=problem.objFunc
        self.nv=0; 
        
        self._start(xInit)
        
        p=self.initalPos.size
        kmax = 2 if p <= 2 else min(p, 4)
        
        for _ in range(kmax):
            pos_copy=self.pos.copy()
            self._explore()
            self._move(pos_copy)
        
        self.bestDec=self.pos
        self.bestObj=self.f
        
        return (self.bestDec, self.bestObj)
    
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
        
    def _start(self, xInit):
        
        self.initalPos=xInit
        
        p = xInit.size
        D = 2 ** (np.arange(1, p + 1).reshape(-1, 1) / (p + 2))
        
        self.nv=1
        self.D=D
        self.f=self.func(xInit)
        self.pos=self.initalPos