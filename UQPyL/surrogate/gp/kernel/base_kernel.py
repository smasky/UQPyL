import numpy as np
from typing import Union

from ...setting import Setting
class BaseKernel():
    def __init__(self):
        
        self.setting = Setting()
        self.setting.defaultOwner = "kernel"
        self.heterogeneous = False

    @property
    def displayName(self):
        return getattr(self, "name", self.__class__.__name__)

    def getActiveParameters(self):
        return self.setting.getParaList(owner="kernel", tunableOnly=False)
        
    def __check_array__(self, value: Union[float,np.ndarray]):
        
        if isinstance(value, float):
            value = np.array([value])
        elif isinstance(value, np.ndarray):
            if value.ndim > 1:
                value = value.ravel()
        else:
            raise ValueError("Please make sure the type of value")
        
        return value
    
    def initialize(self, nInput):
        if self.setting.hasPara("l"):
            size = nInput if self.heterogeneous else None
            self.setting.expandParam("l", size=size)
