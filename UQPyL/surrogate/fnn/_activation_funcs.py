import numpy as np

def identity(X: np.ndarray):
    """
     identity nothing to do
    """
    
def tanh(X: np.ndarray):
    np.tanh(X, out=X)

def relu(X: np.ndarray):
    np.maximum(X, 0, out=X)

def leaky_relu(X: np.ndarray):
    X[X<=0]*=0.1

def elu(X: np.ndarray):
    tmp=np.copy(X[X<=0])
    X[X<=0]=0.1*(np.exp(tmp)-1)

def relu6(X: np.ndarray):
    np.maximum(X, 0, out=X)
    np.minimum(X, 6, out=X)

IDENTITY=0
TANH=1
RELU=2
LEAKY_RELU=3
ELU=4
RELU6=5
ACTIVATIONS=[identity, tanh, relu, leaky_relu, elu, relu6]


def identity_dev(Z: np.ndarray, deltas: np.ndarray):
    """
    nothing to do
    """
    

def tanh_dev(Z: np.ndarray, deltas: np.ndarray):
    """
    =1-tanh^2(X)
    """
    
    deltas *= 1 - Z**2
    
def relu_dev(Z: np.ndarray, deltas: np.ndarray):
    
    deltas[Z <= 0] = 0  #Z<0 =0
    
def leaky_relu_dev(Z: np.ndarray, deltas: np.ndarray):
    
    deltas[Z <= 0 ] *= 0.1

def elu_dev(Z: np.ndarray, deltas: np.ndarray):
    tmp=Z[ Z <= 0 ]
    deltas[ Z <= 0 ] *=0.1*np.exp(tmp)

def relu6_dev(Z: np.ndarray, deltas: np.ndarray):
    deltas[ Z <= 0 ]=0
    deltas[ Z >= 6 ]=0
    
DERIVATIVES=[identity_dev, tanh_dev, relu_dev, leaky_relu_dev, elu_dev, relu6_dev]