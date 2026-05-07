import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lu, pinv
from typing import Tuple, Optional, Literal
from copy import deepcopy

from .kernel import BaseKernel, Cubic
from ..base import SurrogateABC
from ..scaler import Scaler
from ..poly import PolyFeature

class RBF(SurrogateABC):
    """
    Radial basis function surrogate model.

    The model interpolates or smooths training data by combining radial basis
    responses with an optional polynomial tail, depending on the selected kernel.

    Examples:
        >>> model = RBF()
        >>> model.fit(xTrain, yTrain)
        >>> yPred = model.predict(xPred)

    References:
        [1] M. D. Buhmann, Radial Basis Functions: Theory and Implementations,
            Cambridge University Press, 2003.
    """
    
    name = "RBF"
     
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None), 
                    polyFeature: PolyFeature = None,
                        kernel: Optional[BaseKernel] = Cubic(), 
                            C_smooth: int = 0.0, 
                            C_smooth_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        """
        :param scalers: Tuple of input and output scalers.
        :param polyFeature: Polynomial features to be used.
        :param kernel: Kernel function for the RBF network.
        :param C_smooth: Smoothing parameter.
        :param C_smooth_attr: Attribute for the smoothing parameter.
        """
        super().__init__(scalers, polyFeature)
        
        self.setting.set("C_smooth", C_smooth, C_smooth_attr)

        self.registerParameterApplier("kernel", self.setKernel)
        self._kernelChoiceRegistered = False
        
        self.kernel = None
        self.setKernel(kernel)

    def _prepare_training_components(self, xTrain: np.ndarray):
        if hasattr(self.kernel, "initialize"):
            self.kernel.initialize(xTrain.shape[1])
        
    def setKernel(self, kernel: BaseKernel):
        """
        Set the kernel function for the RBF network.
        """
        oldKernelNames = []
        if self.kernel is not None:
            oldKernelNames = [
                name for name in self.kernel.setting.getParaList(owner="kernel", tunableOnly=False)
                if name != "kernel"
            ]

        kernelChoiceValue = self.setting.parVal.get("kernel", None)
        kernelChoiceAttr = self.setting.parSet.get("kernel", None)
        kernelChoiceOwner = self.setting.parOwner.get("kernel", None)

        if oldKernelNames:
            self.setting.removeParas(oldKernelNames)

        if not hasattr(kernel, "_templateSetting"):
            kernel._templateSetting = deepcopy(kernel.setting)
        kernel.setting = deepcopy(kernel._templateSetting)
        
        self.kernel = kernel
        self.setting.mergeSetting(self.kernel.setting)
        self.kernel.setting = self.setting

        if kernelChoiceValue is not None and kernelChoiceAttr is not None:
            self.setting.parVal["kernel"] = self.setting._normalize_choice_array(kernel, kernelChoiceAttr)
            self.setting.parSet["kernel"] = kernelChoiceAttr
            self.setting.parType["kernel"] = 2
            self.setting.parOwner["kernel"] = kernelChoiceOwner
            self.setting.parLB["kernel"] = np.asarray([0.0])
            self.setting.parUB["kernel"] = np.asarray([float(len(kernelChoiceAttr[0]))])
            self.setting.parLog["kernel"] = False

        if self.xTrain is not None and hasattr(self.kernel, "initialize"):
            self.kernel.initialize(self.xTrain.shape[1])
        self.resetFitState()
        return self

    def setKernelChoices(self, kernels):
        self.registerChoiceParameter("kernel", kernels, owner="kernel")
        self._kernelChoiceRegistered = True
        return self

    def _get_tail_matrix(self, kernel: BaseKernel, train_X: np.ndarray):
        """
        Get the tail matrix for the RBF network based on the kernel type.
        
        :param kernel: Kernel function used in the RBF network.
        :param train_X: Training input data.
        :return: Tail matrix for the RBF network.
        """
        if kernel.name == "Cubic" or kernel.name == "Thin_plate_spline":
            tail_matrix = np.ones((self.n_samples, self.n_features + 1))
            tail_matrix[:self.n_samples, :self.n_features] = train_X.copy()
            return tail_matrix
        elif kernel.name == "Linear" or kernel.name == "Multiquadric":
            tail_matrix = np.ones((self.n_samples, 1))
            return tail_matrix
        
        else:
            
            return None
    
    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Fit the RBF model to the training data.
        
        :param xTrain: Training input data.
        :param yTrain: Training output data.
        """
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)

        nSample, nFeature = xTrain.shape
        
        C_smooth = self.setting.get("C_smooth")
        
        A_Matrix = self.kernel.get_A_Matrix(xTrain) + C_smooth
        
        P, L, U = lu(a=A_Matrix)
        L = np.dot(P, L)
        degree = self.kernel.get_degree(nFeature)
        
        if degree:
            bias = np.vstack((yTrain, np.zeros((degree, 1))))
        else:
            bias = yTrain
        
        solve = np.dot(np.dot(pinv(U), pinv(L)), bias)

        if degree:
            coe_h = solve[nSample:, :]
        else:
            coe_h = 0
        
        self.fitState["coe_h"] = coe_h
        self.fitState["coe_lambda"] = solve[:nSample, :]
        return self
          
    def predict(self, xPred: np.ndarray, returnStd: bool = False,
                returnVar: bool = False):
        """
        Predict outputs for given input data using the RBF model.
        
        :param xPred: Input data for prediction.
        :return: Predicted output data.
        """
        self._normalize_predict_flags(returnStd, returnVar)
        self.requireFitted("coe_h", "coe_lambda")

        xPred = self.__X_transform__(xPred)
        _, nFeature = xPred.shape
        
        dist = cdist(xPred, self.xTrain)
        temp1 = np.dot(self.kernel.evaluate(dist), self.fitState["coe_lambda"])
        temp2 = np.zeros((temp1.shape[0], 1))
        
        degree = self.kernel.get_degree(nFeature)
        if degree:
            if degree > 1:
                temp2 = temp2 + np.dot(xPred, self.fitState["coe_h"][:-1, :])
            if degree > 0:
                temp2 = temp2 + np.repeat(self.fitState["coe_h"][-1:, :], temp1.shape[0], axis=0)
        
        return self.__Y_inverse_transform__(temp1 + temp2)
