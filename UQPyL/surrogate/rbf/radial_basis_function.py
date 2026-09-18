from .._kernel import installKernel
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lu, pinv
from typing import Tuple, Optional, Literal

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
                        kernel: Optional[BaseKernel] = None,
                            C_smooth: float = 0.0,
                            C_smooth_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        """
        :param scalers: Tuple of input and output scalers.
        :param polyFeature: Polynomial features to be used.
        :param kernel: Kernel function for the RBF network.
        :param C_smooth: Finite, nonnegative smoothing strength applied only to
                         the kernel-block diagonal with the kernel's sign.
        :param C_smooth_attr: Attribute for the smoothing parameter.
        """
        super().__init__(scalers, polyFeature)
        
        self.setting.set("C_smooth", C_smooth, C_smooth_attr)

        self.registerParameterApplier("kernel", self.setKernel)
        
        self.kernel = None
        self.setKernel(Cubic() if kernel is None else kernel)

    def _prepare_training_components(self, xTrain: np.ndarray):
        if hasattr(self.kernel, "initialize"):
            self.kernel.initialize(xTrain.shape[1])
        
    def setKernel(self, kernel: BaseKernel):
        return installKernel(self, kernel, BaseKernel)

    def setKernelChoices(self, kernels):
        self.registerChoiceParameter("kernel", [kernel.clone() for kernel in kernels], owner="kernel")
        return self

    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Fit the RBF model to the training data.
        
        :param xTrain: Training input data.
        :param yTrain: Training output data.
        """
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)

        nSample, nFeature = xTrain.shape
        
        smooth = np.asarray(self.setting.get("C_smooth"), dtype=float)
        if smooth.size != 1 or not np.all(np.isfinite(smooth)) or np.any(smooth < 0):
            raise ValueError("C_smooth must be a finite, nonnegative scalar.")

        A_Matrix = self.kernel.get_A_Matrix(xTrain)
        # Preserve the polynomial tail and its zero constraint block.
        diagonal = np.arange(nSample)
        A_Matrix[diagonal, diagonal] += self.kernel.smoothingSign * smooth.item()
        
        P, L, U = lu(a=A_Matrix)
        L = np.dot(P, L)
        degree = self.kernel.get_degree(nFeature)
        
        if degree:
            bias = np.vstack((yTrain, np.zeros((degree, yTrain.shape[1]))))
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
