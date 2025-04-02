import sys
sys.path.insert(0, '.')

# Import the Sphere benchmark function
from UQPyL.problems.single_objective import Sphere

# Instantiate the problem with 10 input dimensions; other settings use defaults
sphere = Sphere(nInput = 10) #Other settings use default

# Import the SCE-UA optimization algorithm
from UQPyL.optimization.single_objective import SCE_UA

# Instantiate the optimizer with default settings
sce = SCE_UA()

# Run the optimization on the Sphere problem
res = sce.run(sphere)

# Extract the best decision variables and objective values
bestDecs = res.bestDecs
bestObjs = res.bestObjs