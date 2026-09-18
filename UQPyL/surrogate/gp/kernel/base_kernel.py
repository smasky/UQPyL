import numpy as np
from typing import Union

from ...setting import Setting
from ..._kernel import KernelTemplate

class BaseKernel(KernelTemplate):
    def diag(self, X):
        """Bounded-memory fallback for custom kernels; built-in kernels override it."""
        self._validateInputs(X)
        result = np.empty(len(X))
        for start in range(0, len(X), 64):
            block = X[start:start + 64]
            result[start:start + len(block)] = np.diag(self(block))
        return result

    _parameterRules = {"l": (False, False, False), "alpha": (True, False, False),
                       "nu": (True, False, True), "constant": (True, True, False),
                       "sigma": (True, True, False)}

    def _validateInputs(self, first, second=None):
        nInput = self._checkFeatureMatrix(first, "first input")
        if second is not None and self._checkFeatureMatrix(second, "second input") != nInput:
            raise ValueError("Kernel inputs must have the same number of features.")
        self.validateParameters(nInput)

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
        self.validateParameters(nInput, checkBounds=True)
        if self.setting.hasPara("l"):
            size = nInput if self.heterogeneous else None
            self._expandKernelParam("l", size=size)
