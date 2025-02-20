import abc
from typing import Tuple, Optional
import numpy as np

from ..DoE import LHS, Sampler
from ..utility import Scaler, MinMaxScaler
from ..surrogates import Surrogate
from ..problems import ProblemABC as Problem

class SA(metaclass=abc.ABCMeta):
    """
    Abstract base class for Sensitivity Analysis (SA) methods.
    This class provides a common interface and shared functionality for different SA methods.
    """

    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]], 
                 verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the SA base class with optional scalers and flags.

        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data.
        :param verboseFlag: bool - If True, enables verbose mode for logging.
        :param logFlag: bool - If True, enables logging of results.
        :param saveFlag: bool - If True, saves the results to a file.
        """
        
        # Initialize input scaler
        if scalers[0] is None:
            self.xScale = None
        else:
            if not isinstance(scalers[0], Scaler):
                raise TypeError("scaler must be an instance of Scaler or None!")
            self.xScale = scalers[0]
        
        # Initialize output scaler
        if scalers[1] is None:
            self.yScale = None
        else:
            if not isinstance(scalers[1], Scaler):
                raise TypeError("scaler must be an instance of Scaler or None!")
            self.yScale = scalers[1]

        # Set flags for verbosity, logging, and saving
        self.verboseFlag = verboseFlag
        self.logFlag = logFlag
        self.saveFlag = saveFlag
        
        # Initialize settings and results
        self.setting = Setting()
        self.result = Result(self)
        
    def setParameters(self, key, value):
        """
        Set a parameter for the sensitivity analysis.

        :param key: str - The name of the parameter.
        :param value: Any - The value of the parameter.
        """
        self.setting.setParameter(key, value)
    
    def getParaValue(self, *args):
        """
        Retrieve the value of one or more parameters.

        :param args: str - The names of the parameters to retrieve.
        :return: The value(s) of the specified parameter(s).
        """
        return self.setting.getParaValue(*args) 
        
    def setProblem(self, problem: Problem):
        """
        Set the problem instance for the sensitivity analysis.

        :param problem: Problem - The problem instance defining the input and output space.
        """
        self.problem = problem
    
    def record(self, key, xLabels, value):
        """
        Record the sensitivity analysis results.

        :param key: str - The key under which to store the results.
        :param xLabels: list - The labels for the input variables.
        :param value: np.ndarray - The sensitivity indices.
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.result.Si[key] = (xLabels, value)
        
    def __check_and_scale_xy__(self, X, Y):
        """
        Check and scale the input and output data if scalers are provided.

        :param X: np.ndarray - The input data.
        :param Y: np.ndarray - The output data.
        :return: Tuple[np.ndarray, np.ndarray] - The scaled input and output data.
        """
        if not isinstance(X, np.ndarray) and X is not None:
            raise TypeError("X must be an instance of np.ndarray or None!")
         
        if self.xScale:
            X = self.xScale.fit_transform(X)
        
        if not isinstance(Y, np.ndarray) and Y is not None:
            raise TypeError("Y must be an instance of np.ndarray or None!")

        if self.yScale:
            Y = self.yScale.fit_transform(Y)
                  
        return X, Y
    
    def evaluate(self, X):
        """
        Evaluate the problem with the given input data.

        :param X: np.ndarray - The input data.
        :return: np.ndarray - The output data.
        """
        res = self.problem.evaluate(X)
        Y = res['objs']
        return Y
    
    def transform_into_problem(self, problem, X):
        """
        Transform the input data into the problem's input space.

        :param problem: Problem - The problem instance.
        :param X: np.ndarray - The input data.
        :return: np.ndarray - The transformed input data.
        """
        return X * (problem.ub - problem.lb) + problem.lb
    
    @abc.abstractmethod
    def analyze(self, X_sa=None, Y_sa=None):
        """
        Abstract method for performing sensitivity analysis.
        Must be implemented by subclasses.
        """
        pass

class Result():
    """
    Class to store and manage the results of sensitivity analysis.
    """

    def __init__(self, obj):
        """
        Initialize the Result class.

        :param obj: SA - The sensitivity analysis object.
        """
        self.Si = {}
        self.labels = []
        self.firstOrder = obj.firstOrder
        self.secondOrder = obj.secondOrder
        self.totalOrder = obj.totalOrder
        self.sa = obj
    
    def generateHDF5(self):
        """
        Generate a dictionary representation of the results for HDF5 storage.

        :return: dict - The results formatted for HDF5 storage.
        """
        result = {}
        
        for key, value in self.Si.items():
            xLabels = value[0]
            matrix = value[1]
            result.setdefault(key, {})
            result[key]['matrix'] = matrix
            for label, v in zip(xLabels, matrix.ravel()):
                result[key][label] = v
                    
        return result
    
    def __str__(self):
        """
        String representation of the results.

        :return: str - The formatted results as a string.
        """
        res = self.Si
        output = ""
        for key, (variables, values) in res.items():
            output += f"{key}:\n"
            for var, val in zip(variables, values):
                output += f"  {var}: {val:.5f}\n"
            output += '\n'
        return output

class Setting():
    """
    Class to manage the parameter settings of the algorithm.
    """

    def __init__(self):
        """
        Initialize the Setting class.
        """
        self.dict = {}
    
    def keys(self):
        """
        Get the keys of the parameter settings.

        :return: list - The keys of the parameter settings.
        """
        return self.dict.keys()
    
    def values(self):
        """
        Get the values of the parameter settings.

        :return: list - The values of the parameter settings.
        """
        return self.dict.values()
    
    def setParameter(self, key, value):
        """
        Set a parameter value.

        :param key: str - The name of the parameter.
        :param value: Any - The value of the parameter.
        """
        self.dict[key] = value
    
    def getParaValue(self, *args):
        """
        Get the value of one or more parameters.

        :param args: str - The names of the parameters to retrieve.
        :return: The value(s) of the specified parameter(s).
        """
        values = []
        for arg in args:
            values.append(self.dict[arg])
        
        if len(args) > 1:
            return tuple(values)
        else:
            return values[0]