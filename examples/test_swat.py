from SWAT_UQ import SWAT_UQ
def evaluate(attrs):
    
    obj1 = attrs['objs'][1] #TOT N-MEAN
    obj2 = attrs['objs'][2] #TOT P-MEAN
    
    x = attrs['x']
    obj3 = x[0] * x[1] / 10 * 4200 * 57 * (x[3]+0.001) + x[2] * 4000 * 600 * (x[4]+0.001) #x[0]*x[1]/10表示面积 公顷；420为单位面积成本，57为子流域总数；400000为耕地面积，公顷

    return (obj1, obj2, obj3)
    
filePath = "D:\swat_opt\TxtInOut2"
tempPath = "D:\\swat_opt\\temp"
from UQPyL.DoE import LHS

specialParaList = [
    ("GWATW01", "float", "ops", "3_10"),
    ("GWATL01", "float", "ops", "3_11"),
    ("VFSR01", "float", "ops", "2_7"),
    ("GWATI01", "int", "ops", "3_5"),
    ("VFSI01", "int", "ops", "2_5")
]
swatCup = SWAT_UQ(workPath = filePath,
                    paraFileName = "paras_infos.txt",
                    evalFileName = "observed1.txt",
                    tempPath = tempPath,
                    swatExeName = "SWAT_64rel.exe",
                    specialParaList = specialParaList,
                    verboseFlag = True,
                    userObjFunc = evaluate,
                    nInput = 5,
                    nOutput = 3,
                    maxThreads = 1, numParallel = 1)  

from UQPyL.optimization import NSGAII, MOASMO, MOEAD
from UQPyL.surrogates.rbf import RBF
from UQPyL.problems.multi_objective import ZDT1

moead = MOEAD(aggregation='TCH', nPop=10, maxFEs=5000, verboseFreq=1, saveFlag=True)
moead.run(swatCup)