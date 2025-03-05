import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import lu, pinv
from typing import Tuple, Optional, Literal

from .kernel import BaseKernel, Cubic
from ..surrogateABC import Surrogate
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

class RBF(Surrogate):
    """
    Radial Basis Function (RBF) network for surrogate modeling.
    
    This class implements an RBF network, which is a type of artificial neural network
    used for function approximation. It uses radial basis functions as activation functions.
    
    Attributes:
        name (str): Name of the surrogate model.
    
    Methods:
        setKernel: Set the kernel function for the RBF network.
        fit: Fit the RBF model to training data.
        predict: Predict outputs for given input data.
    """
    
    name = "RBF"
     
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None), 
                    polyFeature: PolynomialFeatures = None,
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
        
        self.setting.setPara("C_smooth", C_smooth, C_smooth_attr)
        
        self.kernel = kernel
        
        self.setting.mergeSetting(kernel.setting)
        
    def setKernel(self, kernel: BaseKernel):
        """
        Set the kernel function for the RBF network.
        """
        if self.kernel is not None:
            self.setting.removeSetting(self.kernel.setting) 
        
        self.kernel = kernel
        self.setting.mergeSetting(self.kernel.setting)

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
    
    def _fitPure(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Fit the RBF model to the training data.
        
        :param xTrain: Training input data.
        :param yTrain: Training output data.
        """
        nSample, nFeature = xTrain.shape
        
        C_smooth = self.setting.getVals("C_smooth")
        
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
        
        self.coe_h = coe_h
        self.coe_lambda = solve[:nSample, :]
        self.xTrain = xTrain
            
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Fit the RBF model to the training data.
        
        :param xTrain: Training input data.
        :param yTrain: Training output data.
        """
        xTrain, yTrain = self.__check_and_scale__(xTrain, yTrain)
        self._fitPure(xTrain, yTrain)
          
    def predict(self, xPred: np.ndarray):
        """
        Predict outputs for given input data using the RBF model.
        
        :param xPred: Input data for prediction.
        :return: Predicted output data.
        """
        _, nFeature = xPred.shape
        
        xPred = self.__X_transform__(xPred)
        
        dist = cdist(xPred, self.xTrain)
        temp1 = np.dot(self.kernel.evaluate(dist), self.coe_lambda)
        temp2 = np.zeros((temp1.shape[0], 1))
        
        degree = self.kernel.get_degree(nFeature)
        if degree:
            if degree > 1:
                temp2 = temp2 + np.dot(xPred, self.coe_h[:-1, :])
            if degree > 0:
                temp2 = temp2 + np.repeat(self.coe_h[-1:, :], temp1.shape[0], axis=0)
        
        return self.__Y_inverse_transform__(temp1 + temp2)