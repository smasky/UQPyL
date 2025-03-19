
import sys
sys.path.insert(0, '.')

from UQPyL.optimization.single_objective import ASMO
from UQPyL.utility.scalers import MinMaxScaler
from UQPyL.surrogates.kriging import KRG
from UQPyL.surrogates.kriging.kernel import Guass
from UQPyL.problems.single_objective import Sphere
from UQPyL.optimization.single_objective import GA

srg = KRG( scalers = (MinMaxScaler(0, 1), MinMaxScaler(0, 1)), 
                kernel = Guass(heterogeneous = False) )

ga = GA(maxFEs=10000)

problem = Sphere(nInput=15)

asmo = ASMO(nInit=100, optimizer=ga, surrogate=srg, maxTolerateTimes=100, verboseFlag=True, verboseFreq=1, logFlag=False, saveFlag=True)

res = asmo.run(problem, oneStep=True)