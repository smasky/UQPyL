import numpy as np
import scipy

from typing import Union, Optional, Literal, Tuple, List

from ..surrogateABC import Surrogate
from ._activation_funcs import (ACTIVATIONS, DERIVATIVES, IDENTITY,
                                            RELU, TANH, LEAKY_RELU, ELU, RELU6)
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

def square_error(yTrue: np.ndarray, yPred: np.ndarray, derivative: bool=False):
    
    if not derivative:
        return np.mean(np.square(yTrue-yPred))/2
    else:
        return yPred-yTrue

class FNN(Surrogate):
    '''
    fully_connect_neural_network
    
    '''
    def __init__(self,
                 scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 polyFeature: PolynomialFeatures=None,
                 hidden_layer_sizes: List[int]=[200,100], 
                 activation_functions: Union[Literal['identity', 'tanh', 'relu', 'leaky_relu', 'elu', 'relu6'], List[str]]='relu',
                 learning_rate: float=0.001, solver: Literal['sgd', 'adam', 'lbfgs']='adam',
                 out_activation: Literal['identity']='identity', 
                 loss_func: Literal['square_error']='square_error', alpha: float=1,
                 epoch: int=2000, batch_size: int=10, shuffle: bool=True,
                 no_improvement_count: int=6000,
                 ):
        
        super().__init__(scalers, polyFeature)
        
        self.hidden_layer_sizes=hidden_layer_sizes
        self.layer_number=len(hidden_layer_sizes)+2
        self.solver=solver
        self.out_activation_=out_activation
        self.learning_rate=learning_rate
        self.loss_func=eval(loss_func.lower())
        self.alpha=alpha
        self.epoch=epoch
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.tol=1e-1
        self._no_improvement_count=no_improvement_count
        
        if isinstance(activation_functions, str):
            
            self.activation_functions=[ACTIVATIONS[eval(activation_functions.upper())]]*(self.layer_number-2)
            self.derivative_functions=[DERIVATIVES[eval(activation_functions.upper())]]*(self.layer_number-2)
        
        elif isinstance(activation_functions, list):
            
            self.activation_functions=[ACTIVATIONS[eval(func.upper())] for func in activation_functions]
            self.derivative_functions=[DERIVATIVES[eval(func.upper())] for func in activation_functions]
            
###--------------------------------------public functions-----------------------------------------------###

    def predict(self, xPred: np.ndarray) -> np.ndarray:
        
        xPred=np.array(self.__X_transform__(xPred))
        activations=[xPred]+[None]*(self.layer_number-1)
        self.__forward(activations)
        
        return self.__Y_inverse_transform__(activations[-1])
        
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain=self.__check_and_scale__(xTrain, yTrain)
        xTrain=np.array(xTrain)
        yTrain=np.array(yTrain)
        _, nFeature=xTrain.shape
        
        if(yTrain.shape[1]!=1):
            raise ValueError("only support one output now!")
        
        hidden_layer_sizes=self.hidden_layer_sizes
        
        ###############Initialization#################
    
        self.nOutput=yTrain.shape[1]
        layer_units=[nFeature]+hidden_layer_sizes+[self.nOutput]
        
        self.layer_number=len(layer_units)
        
        activations=[xTrain]+[None]*(len(layer_units)-1)
        deltas=[None]*(len(activations)-1)
        
        self.coefs_=[]; self.intercepts_=[]
        for i in range(len(layer_units) - 1):
            coef_init, intercept_init = self.__init_coef(
                layer_units[i], layer_units[i + 1], np.float32
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)
           
        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=xTrain.dtype)
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]

        intercept_grads = [
            np.empty(n_fan_out_, dtype=xTrain.dtype) for n_fan_out_ in layer_units[1:]
        ]
        ########################################
        if self.solver in ['adam']:
            self.__solver_gradient_descent(xTrain, yTrain, activations, deltas,
                                       coef_grads, intercept_grads, layer_units)
        
        elif self.solver in ['lbfgs']:
            self.__solver_lbfgs(xTrain, yTrain, activations, deltas,
                                    coef_grads, intercept_grads, layer_units)
            
#####------------------------------private functions----------------------------------------####     

    def __solver_gradient_descent(self, xTrain: np.ndarray, yTrain: np.ndarray,
                                  activations: List, deltas: List, coef_grads: List,
                                  intercept_grads: List, layer_units: List):
        #################################
        self.best_loss_=np.inf
        no_improvement_count=0
        self.n_iter_=0
        self.loss_curve_=[]
        
        params=self.coefs_+self.intercepts_
        
        if self.solver=='adam':
            self.optimizer_=Adam(params, learning_rate=self.learning_rate, beta_1= 0.9, beta_2=0.999, epsilon=1e-8)
            
        nSample, _=xTrain.shape
        sample_idx=np.arange(nSample, dtype=np.int32)
        
        batch_size=min(self.batch_size, nSample)
        
        ###############iteration############
        for _ in range(self.epoch):
            if self.shuffle:
                sample_idx=np.arange(nSample, dtype=np.int32)
                np.random.shuffle(sample_idx)
            
            accumulated_loss=0.0
            for batch_slice in self.__gen_batch(nSample, batch_size):
                if self.shuffle:
                    X_batch = xTrain[sample_idx[batch_slice],:]
                    Y_batch = yTrain[sample_idx[batch_slice]]
                else:
                    X_batch = xTrain[batch_slice,:]
                    Y_batch = yTrain[batch_slice]
                
                activations[0]=X_batch
                
                batch_loss, coef_grads, intercept_grads = self.__backprop(
                        X_batch,
                        Y_batch,
                        activations,
                        deltas,
                        coef_grads,
                        intercept_grads
                    )
                accumulated_loss += batch_loss * (
                        batch_slice.stop - batch_slice.start
                    )
                
                grads = coef_grads + intercept_grads
                self.optimizer_.update_params(params, grads)
            
            self.n_iter_ += 1
            loss = accumulated_loss / xTrain.shape[0]
            
            self.loss_curve_.append(loss)
            
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]
            
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                no_improvement_count += 1
            else:
                no_improvement_count=0
            
            if no_improvement_count>self._no_improvement_count:
                break
            
    def __solver_lbfgs(self, xTrain: np.ndarray, yTrain: np.ndarray,
                            activations: List, deltas: List, coef_grads: List,
                            intercept_grads: List, layer_units: List):
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0
        
        for i in range(self.layer_number - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.layer_number - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end
            
        packed_coef_inter =self.__pack(self.coefs_, self.intercepts_)

        iprint = -1
        max_fun=15000
        max_iter=20000
        tol=1e-4
        opt_res = scipy.optimize.minimize(
            self.__loss_grad_lbfgs,
            packed_coef_inter,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxfun": max_fun,
                "maxiter": max_iter,
                "iprint": iprint,
                "gtol": tol,
            },
            args=(xTrain, yTrain, activations, deltas, coef_grads, intercept_grads),
        )
        
        self.loss_ = opt_res.fun
        self.__unpack(opt_res.x)
    
    def __loss_grad_lbfgs(self, packed_coef_inter, X, y, 
                         activations, deltas, coef_grads, intercept_grads):
        '''
        
        '''
        self.__unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self.__backprop(
            X, y, activations, deltas, coef_grads, intercept_grads
        )
        grad = self.__pack(coef_grads, intercept_grads)
        return loss, grad
    
    def __compute_loss_grad(self, layer_num: int, nSample: int,activations: List, deltas: List, coef_grads: List, intercept_grads: List):
        
        coef_grads[layer_num] = activations[layer_num].T@deltas[layer_num]
        coef_grads[layer_num] += self.alpha * self.coefs_[layer_num]
        coef_grads[layer_num] /= nSample
        
        intercept_grads[layer_num] = np.mean(deltas[layer_num], 0)
            
    def __backprop(self, X_batch: np.ndarray, Y_batch: np.ndarray, activations: List, deltas: List, coef_grads: List, intercept_grads: List):
        
        nSample, _ =X_batch.shape
        self.__forward(activations)
        
        loss=self.loss_func(Y_batch, activations[-1])
        
        ###L2 Regularization
        l2=0
        for coe in self.coefs_:
            coe=coe.ravel()
            l2+=np.dot(coe,coe)
            
        loss+=(0.5*self.alpha)*l2/nSample
        ####using square erro
        deltas[-1]=self.loss_func(Y_batch, activations[-1], derivative=True)
        
        self.__compute_loss_grad(
            len(activations)-2, nSample, activations, deltas, coef_grads, intercept_grads
        )
        
        derivative_func=self.derivative_functions
        
        for i in range(len(activations) - 2, 0, -1):
            deltas[i - 1] = deltas[i]@self.coefs_[i].T
            derivative_func[i-1](activations[i], deltas[i - 1])
            self.__compute_loss_grad(
                i - 1, nSample, activations, deltas, coef_grads, intercept_grads
            )
        return loss, coef_grads, intercept_grads
    
    def __forward(self, activations: List):
        
        activations_functions=self.activation_functions
        
        for i in range(len(activations)-2):
            activations[i+1]=activations[i]@self.coefs_[i]+self.intercepts_[i]
            activations_functions[i](activations[i+1])
            
        activations[-1]=activations[-2]@self.coefs_[-1]+self.intercepts_[-1]
        output_activation = ACTIVATIONS[eval(self.out_activation_.upper())]
        output_activation(activations[-1])
        
        return activations
##############################tool##########################
    def __pack(self, coefs_, intercepts_):
        """Pack the parameters into a single vector."""
        return np.hstack([l.ravel() for l in coefs_ + intercepts_])
    
    def __unpack(self, packed_parameters):
        """Extract the coefficients and intercepts from packed_parameters."""
        for i in range(self.layer_number - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)

            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]
    
    def __gen_batch(self, nSample, batch_size, min_batch_size=0):
                
        start = 0
        for _ in range(int(nSample // batch_size)):
            end = start + batch_size
            if end + min_batch_size > nSample:
                continue
            yield slice(start, end)
            start = end
        if start < nSample:
            yield slice(start, nSample)
            
    def __init_coef(self, fan_in, fan_out, dtype):
    # Use the initialization method recommended by
    # Glorot et al.
        factor = 6.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = np.random.uniform(
            -init_bound, init_bound, (fan_in, fan_out)
        )
        # coef_init=np.ones((fan_in, fan_out), dtype=np.float32)/2
        intercept_init = np.random.uniform(-init_bound, init_bound, fan_out)
        # intercept_init=np.ones(fan_out, dtype=np.float32)/2
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init
#################################################################

class Adam():
    """Adam Algorithm
    Parameters
    -------------------------
    params
    """
    def __init__(self, params: list, learning_rate: float=0.001, 
                    beta_1: float=0.9, beta_2: float=0.999, epsilon: float=1e-8):
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]
        self.learning_rate_init=learning_rate
        
    def update_params(self, params, grads):
        """Update parameters with given gradients

        Parameters
        ----------
        params : list of length = len(coefs_) + len(intercepts_)
            The concatenated list containing coefs_ and intercepts_ in MLP
            model. Used for initializing velocities and updating params

        grads : list of length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        updates = self._get_updates(grads)
        for param, update in zip((p for p in params), updates):
            param += update
##########################Private Function################################  
    def _get_updates(self, grads):
        """Get the values used to update params with given gradients

        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params

        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [
            self.beta_1 * m + (1 - self.beta_1) * grad
            for m, grad in zip(self.ms, grads)
        ]
        self.vs = [
            self.beta_2 * v + (1 - self.beta_2) * (grad**2)
            for v, grad in zip(self.vs, grads)
        ]
        self.learning_rate = (
            self.learning_rate_init * np.sqrt(1 - self.beta_2**self.t)
            / (1 - self.beta_1**self.t)
        )
        updates = [
            -self.learning_rate * m / (np.sqrt(v) + self.epsilon)
            for m, v in zip(self.ms, self.vs)
        ]
        return updates
    
    
               
            
                
            
        
        
        
        
        
        
        
        