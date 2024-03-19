'''
Author: smasky (Mengtian Wu Hohai Unversity)
Date: 2023-09-05 19:55:09
LastEditTime: 2023-09-05 20:14:22
LastEditors: smasky
Description: 
FilePath: \UQPYL2\Test\Test_practical_problem.py
You will never know unless you try
'''
import sys
import os 
os.chdir('./Test/')
sys.path.append("..")

from UQPyL.Problems import PracticalProblem
import numpy as np

testProblem=PracticalProblem()
def f(a,b):
    return a+b

testProblem.set_func(f)
testProblem.evaluation(1)
