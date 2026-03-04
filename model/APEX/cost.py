import numpy as np

def calc_cost(X):
    return np.sum(X)

def transform_ser(tn):
    return tn * 0.0001

def transform_param(X):
    return X * 1