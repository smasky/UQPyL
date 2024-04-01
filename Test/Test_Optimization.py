import sys
import numpy as np
sys.path.append(".")
from UQPyL.Problems import Sphere, ZDT1
from UQPyL.Experiment_Design import LHS
from UQPyL.Optimization import SCE_UA, ASMO, NSGAII, MOASMO
from UQPyL.Surrogates import RBF, MO_Surrogates, KRG
from UQPyL.Surrogates.RBF_Kernel import Cubic
import matplotlib.pyplot as plt
import os
os.chdir('./Test')

##SCE_UA
# lhs=LHS('center')
# problem=Sphere(dim=10, ub=100, lb=-100)
# Algorithm=SCE_UA(problem)
# Algorithm.run()

##ASMO
# rbf=RBF(kernel=Cubic())
# lhs=LHS('center')
# problem=Sphere(dim=10, ub=100, lb=-100)
# algorithm=ASMO(problem, rbf)
# algorithm.run(maxFE=100)

#NSGAII
# problem=ZDT1(15,2)
# algorithm=NSGAII(problem, NInit=500)
# Xpop,Ypop=algorithm.run()
# plt.scatter(Ypop[:,0],Ypop[:,1])
# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.show()

##MOASMO
# problem=ZDT1(15,2)
# # rbf1=RBF(kernel=Cubic())
# # rbf2=RBF(kernel=Cubic())
# krg1=KRG(np.random.random(15),np.ones(15)*1e4,np.ones(15)*1e-4, regression='poly1', kernel='gaussian', optimizer='Boxmin', fitMode='likelihood')
# krg2=KRG(np.random.random(15),np.ones(15)*1e4,np.ones(15)*1e-4, regression='poly1', kernel='gaussian', optimizer='Boxmin', fitMode='likelihood')
# surrogates=MO_Surrogates(N_Surrogates=2, Models_list=[krg1, krg2])
# algorithm=MOASMO(problem, surrogates, advance_infilling=True)
# Xpop,Ypop=algorithm.run(maxFE=300)# plt.scatter(Ypop[:,0],Ypop[:,1])
Ypop=np.loadtxt('./YY.txt')
plt.scatter(Ypop[:,0],Ypop[:,1])
plt.xlim((0,1))
plt.ylim((0,1))
plt.show()
