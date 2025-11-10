import sys
sys.path.insert(0, '.')

from UQPyL.util import plot_op_curve, plot_op_curve_stat, plot_op_pareto, plot_sa, plot_surrogate

# source = {
#     "GA": "./Result/Data/GA_Sphere_D30_M1_1.nc",
#     "PSO": "./Result/Data/PSO_Sphere_D30_M1_1.nc",
#     "DE": "./Result/Data/DE_Sphere_D30_M1_1.nc",
#     "ABC": "./Result/Data/ABC_Sphere_D30_M1_1.nc",
# }

# plot_op_curve(source = source, yLog = True, ySmooth = True, markevery = 40, xlim = [0, 1000], xMajorLocator = 100)

# folder = "./Result/Data"
# prefix = "GA_Sphere_D30_M1"
# plot_op_curve_stat(folder = folder, prefix = prefix, xCoord = "iter",
#                     yLog = True, ySmooth = True, ci = "t95",
#                     xlim = [0, 1000], xMajorLocator = 100)

# two-objective problems
# from UQPyL.problem import ZDT1

# zdt = ZDT1(10)

# optima = zdt.getPF()

# filepath = "./Result/Data/NSGAII_ZDT1_D30_M2_1.nc"

# plot_op_pareto(filepath = filepath, optima = optima, xlim = [0, 1], xMajorLocator = 0.1, ylim = [0, 1], yMajorLocator = 0.1)

# three-objective problems

# from UQPyL.problem import DTLZ7

# dtlz = DTLZ7(10)

# optima = dtlz.getPF()

# filepath = "./Result/Data/NSGAII_DTLZ7_D10_M3_1.nc"

# plot_op_pareto(filepath = filepath, optima = optima, xlim = [0, 1], xMajorLocator = 0.1, ylim = [0, 1], yMajorLocator = 0.1)

# source = {
#     "FAST": "./Result/Data/FAST_Problem_D8_M1_1.nc",
#     "RBDFAST": "./Result/Data/RBDFAST_Problem_D8_M1_1.nc",
#     "Sobol" : "./Result/Data/Sobol_Problem_D8_M1_1.nc",
#     "MARS" : "./Result/Data/MARS_Problem_D8_M1_1.nc",
# }

# plot_sa(source = source, fontsize = 20)

# from UQPyL.problem import Sphere
# from UQPyL.doe import LHS

# lhs = LHS()

# problem = Sphere(nInput = 15)

# X = lhs.sample(nt = 800, problem = problem)
# Y = problem.objFunc(X)

# XTest = lhs.sample(nt = 100, problem = problem)
# YTest = problem.objFunc(XTest)

# from UQPyL.surrogate.rbf import RBF

# rbf = RBF()

# rbf.fit(X, Y)

# YPred = rbf.predict(XTest)

# plot_surrogate(name = "RBF", yPred = YPred, yTrue = YTest, ylim = [20000, 90000], yMajorLocator = 10000)


# from UQPyL.util import plot_infer_trace

# filepath = "./Result/Data/MH_Gaussian4_D4_M1_1.nc"

# plot_infer_trace(filepath = filepath, fontsize = 20, burnIn = 100)


# from UQPyL.util import plot_infer_stat, plot_infer_stat_combined

# filepath = "./Result/Data/MH_Gaussian4_D4_M1_1.nc"

# plot_infer_stat(filepath = filepath, fontsize = 20, burnIn = 100)

# plot_infer_stat_combined(filepath = filepath, fontsize = 20, burnIn = 100, showCI = True, CI = 0.95)


