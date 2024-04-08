#Delta test
import numpy as np
from ..Experiment_Design import LHS
from ..Optimization import Binary_GA
from scipy.spatial.distance import cdist

lhs=LHS('center')
class DELTA_TEST():
    def __init__(self, problem, NSample=1000, n_neighbors=2):
        self.lb=problem.lb; self.ub=problem.ub
        self.NInput=problem.dim
        self.evaluator=problem.evaluate
        self.n_neighbors=n_neighbors
        
        self.NSample=NSample
    
    def analyze(self):
        '''
        main procedure
        '''
        lb=self.lb
        ub=self.ub
        NInput=self.NInput
        NSample=self.NSample
        
        X_seq=(ub-lb)*lhs(NSample, NInput)+lb
        Y_seq=self.evaluator(X_seq)
        
        self.X_seq=X_seq; self.Y_seq=Y_seq
        bin_ga=Binary_GA(self.cal_delta, NInput)
        a=bin_ga.run()    
        b=1
    
    def cal_delta(self, exclude_feature_list):
        X=np.copy(self.X_seq)
        y=np.copy(self.Y_seq)
        exclude_feature = [index for index, value in enumerate(exclude_feature_list) if value == 1]
        if exclude_feature is not None:
            # 排除指定的特征
            X = np.delete(X, exclude_feature, axis=1)
        
        # 计算所有样本点之间的距离
        distances = cdist(X, X)
        np.fill_diagonal(distances, np.inf)
        
        # 为每个样本找到最近的n个邻居
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        # 计算Delta值
        deltas = []
        for i in range(len(X)):
            neighbor_deltas = (y[i] - y[neighbors_indices[i]])**2
            delta = np.mean(neighbor_deltas)
            deltas.append(delta)
        
        # 返回平均Delta值
        return np.mean(deltas) 

        