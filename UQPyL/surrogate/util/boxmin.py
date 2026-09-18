import numpy as np
class Boxmin():
    """
    Internal MP optimizer used by surrogate models.

    Notes:
        - This is the current concrete implementation of the "MP" family
          used by GPR/KRG internal hyper-parameter optimization.
        - It is a single-point bounded local search routine.
        - Multiplicative steps operate on a positive internal interval [1, 2].
          Objective evaluations and returned decisions use the supplied bounds.
        - Stop when multiplicative step factors are within 0.001 of one.
    """
    type = "MP"
    name = "Boxmin"
    
    def __init__(self) -> None:
        
        pass
        
    ###################################Interface Function#################################
    def run(self, problem, xInit=None, seed=None):
        lower = np.asarray(problem.lb, dtype=float).ravel().copy()
        upper = np.asarray(problem.ub, dtype=float).ravel().copy()
        if lower.size != problem.nInput or upper.size != problem.nInput or not lower.size:
            raise ValueError("Boxmin bounds must match problem.nInput.")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
            raise ValueError("Boxmin requires finite, ordered bounds.")
        with np.errstate(over="ignore"):
            span = upper - lower
        if not np.all(np.isfinite(span)):
            raise ValueError("Boxmin requires finite bound widths.")

        rng = np.random.default_rng(seed)
        if xInit is None:
            xInit = rng.uniform(lower, upper)
        else:
            xInit = np.asarray(xInit, dtype=float).ravel().copy()
            if xInit.size != lower.size or not np.all(np.isfinite(xInit)):
                raise ValueError("xInit must contain one finite value per input.")
        xInit = np.clip(xInit, lower, upper)

        active = span > 0
        self.lb = np.ones(lower.size)
        self.ub = np.where(active, 2.0, 1.0)
        position = np.zeros(lower.size)
        np.divide(xInit - lower, span, out=position, where=active)
        position = np.clip(1.0 + position, self.lb, self.ub)

        def toParameters(pos):
            # Clipping also protects original bounds from roundoff on decoding.
            return np.clip(lower + (pos - 1.0) * span, lower, upper)

        def evaluate(pos):
            self.nv += 1
            return float(np.asarray(problem.objFunc(toParameters(pos))).item())

        self.func = evaluate
        self.nv = 0
        self._start(position)

        while np.any(self.D[active] > 1.001):
            pos_copy=self.pos.copy()
            self._explore()
            self._move(pos_copy)
        
        self.bestDec=toParameters(self.pos)
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
            if np.array_equal(pos_c, pos):
                break
            ff=self.func(pos_c)
            
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
            if self.lb[k] == self.ub[k]:
                continue
            pos_c=pos.copy()
            DD = float(np.asarray(self.D[k]).reshape(-1)[0])
            
            if pos[k]==self.ub[k]:
                atbd=True
                pos_c[k]=pos[k]/np.sqrt(DD)
            elif pos[k]==self.lb[k]:
                atbd=True
                pos_c[k]=pos[k]*np.sqrt(DD)
            else:
                atbd=False
                pos_c[k]=np.minimum(self.ub[k], pos[k]*DD)
            
            pos_c[k] = np.clip(pos_c[k], self.lb[k], self.ub[k])
            ff=self.func(pos_c)
            
            if ff<f:
                pos=pos_c.copy()
                f=ff
            else:
                if not atbd:
                    pos_c[k] = np.maximum(self.lb[k], pos[k] / DD)
                    ff = self.func(pos_c)
                    if ff<f:
                        pos=pos_c.copy()
                        f=ff
        self.pos=pos
        self.f=f
        
    def _start(self, xInit):
        
        self.initalPos=xInit
        
        p = xInit.size
        D = 2 ** (np.arange(1, p + 1, dtype=float) / (p + 2))
        
        self.D=D
        self.f=self.func(xInit)
        self.pos=self.initalPos
