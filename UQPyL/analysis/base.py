import abc
from typing import Tuple, Optional
import numpy as np

from ..util import Scaler
from ..problem import ProblemABC as Problem

class AnalysisABC(metaclass=abc.ABCMeta):
    
    """
    Abstract base class for analysis methods.
    This class provides some common interfaces for analysis methods.
    """

    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]], 
                 verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        
        """
        Initialize the analysis base class with optional scalers and flags.

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
        
    def setParaValue(self, key, value):
        
        """
        Set a parameter for the sensitivity analysis.

        :param key: str - The name of the parameter.
        :param value: Any - The value of the parameter.
        """
        
        self.setting.setParaValue(key, value)
    
    def getParaValue(self, *args):
        
        """
        Retrieve the value of one or more parameters.

        :param args: str - The names of the parameters to retrieve.
        :return: The value(s) of the specified parameter(s).
        """
        
        return self.setting.getParaValue(*args) 
        
    def setProblem(self, problem: Problem):
        
        """
        Set the problem instance for the analysis.

        :param problem: Problem - The problem instance defining the input and output space.
        """
        
        self.problem = problem
    
    def record(self, target, indicator, labels, values):

        """
        Record the analysis results.
        
        :param target: str - The target of the objective function or constraint function.
        :param indicator: str - The indicator of the analysis.
        :param labels: list - The labels for the input variables.
        :param value: list - The sensitivity indices.
        """
                        
        self.result.res['Results'].setdefault(target, {})
        self.result.res['Results'][target].setdefault(indicator, {})
        
        for label, v in zip(labels, values):
            
            self.result.res['Results'][target][indicator][label] = v
        
        self.result.res['Results'][target][indicator]['array'] = np.array(values)
    
    def check_Y(self, Y, target = 'objFunc', index = 'all'):
        # Evaluate the problem if Y is not provided
        if Y is None:
            if target == 'objFunc':
                Y = self.problem.objFunc(X)
            elif target == 'conFunc':
                Y = self.problem.conFunc(X)
            else:
                raise ValueError("Target must be 'objFunc' or 'conFunc'!")

        if index != 'all':
            if not isinstance(index, list):
                raise ValueError("Index must be a list of integers!")
            else:
                try:
                    Y = Y[:, index]
                except:
                    raise ValueError("Please check the index you set!")
        
        return Y
    
    def saveHistory(self, X, Y):
        
        self.result.res['History'] = {
            'methodName' : self.name, 
            'problemName' : self.problem.name, 
            'X' : X, 
            'Y' : Y, 
            'settings' : self.setting.dict
            }
 
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

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        if self.yScale:
            Y = self.yScale.fit_transform(Y)
                  
        return X, Y
    
    def evaluate(self, X, target = 'objFunc'):
        
        """
        Evaluate the problem with the given input data.

        :param X: np.ndarray - The input data.
        :param target: str - The target to evaluate.
        :return: np.ndarray - The output data.
        """
        
        if target == 'objFunc':
            Y = self.problem.objFunc(X)
        elif target == 'conFunc':
            Y = self.problem.conFunc(X)
        else:
            raise ValueError("Target must be 'objFunc' or 'conFunc'!")
        
        return Y
    
    @abc.abstractmethod
    def analyze(self, X = None, Y = None):
        
        """
        Abstract method for performing analysis.
        Must be implemented by subclasses.
        """
        
        pass

class Result():
    
    """
    Class to store and manage the results of analysis.
    """

    def __init__(self, obj):
        """
        Initialize the Result class.

        :param obj: The analysis object.
        """
        
        self.res = {
            'History': {}, 
            'Results': {}
            }
        
        self.obj = obj
    
    def generateHDF5(self):
        """
        Generate a dictionary representation of the results for HDF5 storage.

        :return: dict - The results formatted for HDF5 storage.
        """

        return self.res
    
    # def __str__(self):
    #     """
    #     String representation of the results.

    #     :return: str - The formatted results as a string.
        
    #     """
        
    #     res = self.res
    #     output = ""
        
    #     for key, (labels, value) in res.items():
    #         output += f"{key}:\n"
    #         for label, v in zip(labels, value.ravel()):
    #             output += f"  {label}: {v:.5f}\n"
    #         output += '\n'
            
    #     return output

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
    
    def setParaValue(self, key, value):
        
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