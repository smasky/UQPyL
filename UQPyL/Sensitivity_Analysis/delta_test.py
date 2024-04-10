#Delta test
import numpy as np
from ..Experiment_Design import LHS
from ..Optimization import Binary_GA
from .      .Utility import MinMaxScaler
from scipy.spatial.distance import cdist

lhs=LHS('center')
scale=MinMaxScaler(0,1)
class DELTA_TEST():
    def __init__(self, problem, NSample=1000, n_neighbors=2, surrogate=None, NSurrogate=50, XInit=None, YInit=None):
        self.lb=problem.lb; self.ub=problem.ub
        self.NInput=problem.dim
        self.evaluator=problem.evaluate
        self.n_neighbors=n_neighbors
        
        self.NSample=NSample
        
        self.YInit=YInit
        if self.surrogate:
            if XInit is None:
                self.XInit=lhs(NSurrogate, self.dim)*(self.ub-self.lb)+self.lb
                
    def analyze(self):
        '''
        main procedure
        '''
        lb=self.lb
        ub=self.ub
        NInput=self.NInput
        NSample=self.NSample
        
        X_seq=(ub-lb)*lhs(NSample, NInput)+lb
        
        if self.surrogate:
            if self.XInit is None:
                self.XInit=lhs(self.NSurrogate, self.dim)*(self.ub-self.lb)+self.lb
                self.YInit=self.evaluate(self.XInit)
            self.surrogate.fit(self.XInit, self.YInit)
            Y_seq=self.surrogate.predict(X_seq)
        else:
            Y_seq=self.evaluate(X_seq)
            
        Y_seq_scale=scale.fit_transform(Y_seq)
        
        self.X_seq=X_seq; self.Y_seq=Y_seq_scale
        bin_ga=Binary_GA(self.cal_delta, NInput)
        best_paras, best_value, history_paras, _=bin_ga.run()
        
        idx=[index for index, value in enumerate(best_paras) if value == 1]

        score=np.sum(history_paras, axis=0)
        
        return idx, score[0, idx]
    
    def cal_delta(self, exclude_feature_list):
        X=np.copy(self.X_seq)
        y=np.copy(self.Y_seq)
        exclude_feature = [index for index, value in enumerate(exclude_feature_list) if value == 0]
        if exclude_feature is not None:
            X = np.delete(X, exclude_feature, axis=1)
            
        distances = cdist(X, X)
        np.fill_diagonal(distances, np.inf)
        
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        deltas = []
        for i in range(len(X)):
            neighbor_deltas = (y[i] - y[neighbors_indices[i]])**2
            delta = np.mean(neighbor_deltas)
            deltas.append(delta)
        
        return np.mean(deltas) 

        