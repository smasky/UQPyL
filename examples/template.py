import sys
sys.path.append('.')

import os
uqPath=os.path.dirname("../UQPyL/")
sys.path.insert(0, uqPath)

import numpy as np

from UQPyL.problems import PracticalProblem

def evaluate(X):
    # 目标值计算函数
    pass
    #Return Y
    # X 是输入，应为numpy.2d矩阵输入，X的行数为样本数，列数为自变量数
    # Y 是输出，应为numpy.2d矩阵输出，Y的行数为样本数，列数未目标数
    
def constraint(X):
    # 约束条件计算函数
    pass
    #Return C
    # X 是输入，应为numpy.2d矩阵输入，X的行数为样本数，列数为自变量数
    # C 是输出，应为numpy.2d矩阵输出，Y的行数为样本数，列数为约束数量
    
#定义优化问题
N = 10 # 自变量个数
M = 1 #
ub = 100*np.ones(N) # 自变量上限
lb = 0*np.zeros(N) #自变量下线
problem =PracticalProblem(objFunc=evaluate, conFunc=constraint, nInput= N, nOutput = M, ub = ub, lb =lb)