import numpy as np
from typing import Union

from ...setting import Setting
class BaseKernel():
    name = None
    
    def __init__(self, heterogeneous: bool,
                 theta: Union[float, np.ndarray], 
                 theta_attr: Union[dict, None] = None,
                 ):
        
        self.setting = Setting()
        self.setting.defaultOwner = "kernel"
        
        self.heterogeneous = heterogeneous
        
        self.setting.set("theta", theta, theta_attr)

    @property
    def displayName(self):
        return self.name or self.__class__.__name__

    def getActiveParameters(self):
        return self.setting.getParaList(owner="kernel", tunableOnly=False)
        
    def initialize(self, nInput):
        size = nInput if self.heterogeneous else None
        self.setting.expandParam("theta", size=size)
