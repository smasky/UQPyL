import numpy as np
from math import comb

class PolynomialFeatures():
    def __init__(self, degree: int=2, include_bias: bool=False, interaction_only: bool=False):
        self.degree=degree
        self.include_bias=include_bias
        self.interaction_only=interaction_only
    
    def transform(self, trainX: np.ndarray):
        
        n_samples, n_features=trainX.shape
        
        if self.include_bias:
            n_output_features=1
        else:
            n_output_features=0
        
        if self.interaction_only:
            for d in range(1, self.degree+1):
                n_output_features+=comb(n_features, d)
        else: 
            for d in range(1, self.degree+1):
                n_output_features+=comb(d+(n_features-1),n_features-1)
                    
        outTrainX=np.zeros((n_samples, n_output_features))
        
        ######################bias########################
        if self.include_bias:
            outTrainX[:, 0]=np.ones(n_samples)
            current_col=1
        else:
            current_col=0
        #####################degree1#####################
        outTrainX[:, current_col : current_col + n_features] = trainX
        index = list(range(current_col, current_col + n_features))
        current_col += n_features
        index.append(current_col)
        #####################degree>2####################
        for _ in range(2, self.degree + 1):
            new_index = []
            end = index[-1]
            for feature_idx in range(n_features):
                start = index[feature_idx]
                new_index.append(current_col)
                if self.interaction_only:
                    start += index[feature_idx + 1] - index[feature_idx]
                next_col = current_col + end - start
                if next_col <= current_col:
                    break
                np.multiply(
                    outTrainX[:, start:end],
                    trainX[:, feature_idx : feature_idx + 1],
                    out=outTrainX[:, current_col:next_col]
                )
                current_col = next_col
            new_index.append(current_col)
            index = new_index
            
        return outTrainX
        