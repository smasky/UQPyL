import numpy as np
from typing import Union

from .problem_ABC import ProblemABC
###################Basic Test Function##################
#Reference: 
#Xin Yao; Yong Liu; Guangming Lin (1999).Evolutionary programming made faster. , 3(2), 0–102.doi:10.1109/4235.771163
###############################################################
class Sphere(ProblemABC):
    '''
    Types:
        Single Optimization Unimodal
        
    F1->Sphere Function:
        F= \sum x_i
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*100;LB->np.ones(1,30)*-100
     
    Optimal:
        X*=0 0 0 ... 0
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =100,lb: Union[int,float,np.ndarray] =-100,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
    
    def evaluate(self, X: np.ndarray, unit=False) -> np.ndarray:
        '''
            Parameters:
                X: np.ndarray
                    the input data
                unit: bool, default=False
                    whether to transform X to the bound
        '''
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        F=np.sum(X**2,axis=1).reshape(-1,1)      
        
        return F

class Schwefel_2_22(ProblemABC):
    '''
    Types:
        Single Optimization Unimodal
        
    F2-> Schwefel_2_22 Function:
        F= \sum \left | x_i \right |+ \prod \left | x_i \right |
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*10;LB->np.ones(1,30)*-10
     
    Optimal:
        X*=0 0 0 ... 0
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =10,lb: Union[int,float,np.ndarray] =-10,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        F=np.sum(np.abs(X),axis=1).reshape(-1,1)+np.prod(np.abs(X),axis=1).reshape(-1,1)
        return F

class Schwefel_1_22(ProblemABC):
    '''
    Types:
        Single Optimization Unimodal
        
    F3-> Schwefel_1_22 Function:
        F= \sum_{i}  \left ( \sum_{j}^{i} x_j \right )^2
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*100;LB->np.ones(1,30)*-100
     
    Optimal:
        X*=0 0 0 ... 0
        F*=0
    '''
    
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =100,lb: Union[int,float,np.ndarray] =-100,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        F=0
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        X=X**2
        for d in range(self.n_input):
            F+=np.sum(X[:,:d],axis=1).reshape(-1,1)
        return F
    
class Schwefel_2_21(ProblemABC):
    '''
    Types:
        Single Optimization Unimodal
        
    F4-> Schwefel_1_22 Function:
        F= \max_{i} \left | x_i \right | 
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*100;LB->np.ones(1,30)*-100
     
    Optimal:
        X*=0 0 0 ... 0
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =100,lb: Union[int,float,np.ndarray] =-100,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        X=np.abs(X)
        F=np.max(X,axis=1).reshape(-1,1)
        return F
    
class Rosenbrock(ProblemABC):
    '''
    Types:
        Single Optimization Unimodal
        
    F5-> Rosenbrock Function:
        F= \sum \left ( 100\left ( x_{i+1} - x_i^2 \right ) ^2 - \left ( x_i-1 \right ) ^2 \right )
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*30;LB->np.ones(1,30)*-30
     
    Optimal:
        X*=1 1 1 ... 1
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =30,lb: Union[int,float,np.ndarray] =-30,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))    
        Temp1=100*np.square(X[:,1:]-np.square(X[:,:-1]))
        Temp2=np.square(X[:,:-1]-1)
        F=np.sum(Temp1+Temp2,axis=1).reshape(-1,1)
        return F

class Step(ProblemABC):
    '''
    Types:
        Single Optimization Unimodal
        
    F6-> Step Function:
        F= \sum \left ( \left \lfloor x_i+0.5  \right \rfloor \right ) ^2
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*100;LB->np.ones(1,30)*-100
     
    Optimal:
        X*=1 1 1 ... 1
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =100,lb: Union[int,float,np.ndarray] =-100,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))     
        F=np.sum(np.square(np.floor(X)+0.5),axis=1).reshape(-1,1)
        
        return F
    
class Quartic(ProblemABC):
    '''
    Types:
        Single Optimization Unimodal
        
    F7-> Quartic Function:
        F= \sum \left ( ix_i^4 \right )+random \left [ 0 , 1 \right )
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*1.28;LB->np.ones(1,30)*-1.28
     
    Optimal:
        X*=1 1 1 ... 1
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =1.28,lb: Union[int,float,np.ndarray] =-1.28,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))  
        Temp=np.linspace(1,self.n_input,self.n_input)*np.power(X,4)
        F=np.sum(Temp,axis=1).reshape(-1,1)+np.random.random((Temp.shape[0],1))          
        return F

class Schwefel_2_26(ProblemABC):
    '''
    Types:
        Single Optimization Multimodal
        
    F8-> Quartic Function:
        F= \sum \left( x_i sin\left ( \sqrt{ \left | x_i \right |} \right ) \right)
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*500;LB->np.ones(1,30)*-500
     
    Optimal:
        X*=420.9687 420.9687 ... 420.9687
        F*=-12569.5
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =500,lb: Union[int,float,np.ndarray] =-500,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X)) 
        Temp=np.sin(np.sqrt(np.abs(X)))*X
        F=np.sum(Temp,axis=1).reshape(-1,1)*-1         
        return F
    
class Rastrigin(ProblemABC):
    '''
    Types:
        Single Optimization Multimodal
        
    F9-> Rastrigin Function:
        F= \sum \left [ x_i^2 - 10 cos\left ( 2 \pi x_i   +10 \right ) \right ]
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*5.12;LB->np.ones(1,30)*-5.12
     
    Optimal:
        X*=0 0 0 ... 0
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =5.12,lb: Union[int,float,np.ndarray] =-5.12,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        F=np.sum(np.square(X)-10*np.cos(2*np.pi*X)+10,axis=1).reshape(-1,1)
        return F

class Ackley(ProblemABC):
    '''
    Types:
        Single Optimization Multimodal
        
    F10-> Ackley Function:
        F= -20 \exp \left ( -0.2 \sqrt{ \frac{1}{n_input} \sum x_i^2 } \right ) 
                  - \exp \left ( \frac{1}{n_input} \sum cos 2 \pi x_i \right ) + 20 + e
     
    Default setting:
        Dims->30;Ub->np.ones(1,30)*32;LB->np.ones(1,30)*-32
     
    Optimal:
        X*=0 0 0 ... 0
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =32,lb: Union[int,float,np.ndarray] =-32,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
    
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X)) 
        Temp1=np.exp(np.sqrt(np.sum(np.square(X),axis=1)/self.n_input)*-0.2)*-20
        Temp2=np.exp(np.sum(np.cos(2*np.pi*X),axis=1)/self.n_input)*-1+20+np.e  
        F=(Temp1+Temp2).reshape(-1,1)
        return F
    
class Griewank(ProblemABC):
    '''
    Types:
        Single Optimization Multimodal
        
    F11-> Griewank Function:
        F= \frac{1}{4000} \sum x_i^2 - \prod \cos \left ( \frac{x_i}{\sqrt{i}} \right ) + 1
        
    Default setting:
        Dims->30;Ub->np.ones(1,30)*600;LB->np.ones(1,30)*-600
     
    Optimal:
        X*=0 0 0 ... 0
        F*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =600,lb: Union[int,float,np.ndarray] =-600,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
    
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X)) 
        I=np.sqrt(np.atleast_2d(np.linspace(1,self.n_input,self.n_input)))
        F=np.sum(np.square(X), axis=1).reshape(-1,1)/4000-np.prod(np.cos(X/I),axis=1).reshape(-1,1)+1     
        return F

##############Other Common Used Functions##############
####
#########################

class Trid(ProblemABC):
    '''
    Types:
        Single Optimization Multimodal
        
    F12-> Trid Function:
        F= \sum_{i=1}{D} \left ( x_i - 1 \right )^2 - \sum_{i=2}^{D} \left ( x_i x_{i-1} \right ) 
        
    Default setting:
        Dims->30;Ub->np.ones(1,30)*(30^2);LB->np.ones(1,30)*-(30^2)
     
    Optimal:
        X_i^*=i(D+1-i),i=1,2,...,D
        F^*=-D(D+4)(D-1)/6
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =900,lb: Union[int,float,np.ndarray] =-900,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
    
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        F=np.sum(np.square(X-1),axis=1).reshape(-1,1)-np.sum(X[:,1:]*X[:,:-1],axis=1).reshape(-1,1)
        return F

class Bent_Cigar(ProblemABC):
    '''
    Types:
        Single Optimization Multimodal
        
    F13-> Bent Cigar Function:
        F= x_1^2 + 10^6 \sum_{i=2}^{D} x_i^2
        
    Default setting:
        Dims->30;Ub->np.ones(1,30)*10;LB->np.ones(1,30)*-10
     
    Optimal:
        X^*=0 0 0 ...0
        F^*=0
    '''
    def __init__(self, n_input: int =30, ub: Union[int,float,np.ndarray] =10, lb: Union[int,float,np.ndarray] =-10, disc_var=None, cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
    
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        F=((X[:,0]**2)+np.sum(np.square(X[:,1:]),axis=1)*(10**6)).reshape(-1,1)
        return F
    
class Discus(ProblemABC):
    '''
    Types:
        Single Optimization Multimodal
        
    F14-> Discus Function:
        F= 10^6 x_1^2 +  \sum_{i=2}^{D} x_i^2
        
    Default setting:
        Dims->30;Ub->np.ones(1,30)*10;LB->np.ones(1,30)*-10
     
    Optimal:
        X^*=0 0 0 ...0
        F^*=0
    '''
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =10,lb: Union[int,float,np.ndarray] =-10,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
    
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        F=(X[:,0]**2*(10**6)+np.sum(np.square(X[:,1:]),axis=1)).reshape(-1,1)
        return F
    
class Weierstrass(ProblemABC):
    '''
    Types:
        Single Optimization Multimodal
        
    F15-> Weierstrass Function:
        F= \sum_{i=1}^{D} \left ( \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \left( x_i + 0.5 \right) \right) \right )-
                \sum_{k=0}^{k_{max}} a^k \cos \left( \pi b^k \right)
        
    Default setting:
        Dims->30;Ub->np.ones(1,30)*0.5;LB->np.ones(1,30)*-0.5
        k_max=20;a=0.5;b=3
    Optimal:
        X^*=0 0 0 ...0
        F^*=0
    '''
    kMax=20
    a=0.5
    b=3
    def __init__(self, n_input:int =30, ub: Union[int,float,np.ndarray] =0.5,lb: Union[int,float,np.ndarray] =-0.5,disc_var=None,cont_var=None):
        
        super().__init__(n_input,1,ub,lb,disc_var,cont_var)
        
    def evaluate(self, X: np.ndarray, unit: bool=False) -> np.ndarray:
        
        X=self._check_2d(X)
        if unit:
            X=self._unit_X_transform_to_bound(np.atleast_2d(X))
        K=np.atleast_2d(np.linspace(1,self.kMax,self.kMax))
        aK=np.power(self.a,K)
        bK=np.power(self.b,K)
        aK_expand=np.tile(aK.transpose(),(1,self.n_input)).reshape(1,-1)
        bK_expand=np.tile(bK.transpose(),(1,self.n_input)).reshape(1,-1)
        
        
        X_expand=np.tile(X, (1,self.kMax))
        Addition=np.sum(aK*np.cos(bK*np.pi),axis=1).reshape(-1,1)*self.n_input
        F=np.sum(np.cos(2*np.pi*(X_expand+0.5)*bK_expand)*aK_expand,axis=1).reshape(-1,1)-Addition
        return F