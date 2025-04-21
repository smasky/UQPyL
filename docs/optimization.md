# Optimization

## 🫐 What is the optimization

**Optimization** refers to the process of finding the best solution from feasible space, typically by minimizing or maximizing a specific objective function. In the context of modelling and simulation, optimization is used to identify input parameters that yield the most desirable output — for example, the highest fidelity, the lowest cost, or the best performance.

<br>

A optimization problem should include:

- One more **objective functions** to minimize or maximize.
- Clear **Decision Variable** with their attributes, e.g., type, range ...
- **Constraints** (optional) that define the limits or rules for the inputs.

In addition, optimization problems can be categorized as **single-objective** or **multi-objective**, depending on how many objectives need to be optimized.

## 🫛 Overview of `UQPyL.optimization`

Optimization is essential for modeling and solving real-world problems. The `optimization` module in UQPyL provides a collection of widely used algorithms, organized into two submodules: `single_objective` and `multi_objective`.

Below is a list of currently implemented algorithms. Note that this module is actively being updated.

<br>

**Algorithms for single-objective optimization:**

| Abbreviation | Full Name |   Optimization Label   |   References  |
|--------------|-----------| ----------|---------------|
| SCE-UA | Shuffled Complex Evolution| Single | [Duan et al. (1992)](https://link.springer.com/article/10.1007/BF00939380)|
| ML-SCE-UA| M&L Shuffled Complex Evolution| Single | [Muttil, Liong (2006)](https://www.worldscientific.com/doi/abs/10.1142/9789812707208_0036) |
| GA | Genetic Algorithm| Single | [Holland (1992)](https://direct.mit.edu/books/monograph/2574/Adaptation-in-Natural-and-Artificial-SystemsAn)|
| CSA | Cooperation Search Algorithm | Single | [Feng et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1568494620306724) |
| PSO | Particle Swarm Optimization | Single | [Kennedy and Eberhart (1995)](https://ieeexplore.ieee.org/abstract/document/488968/) |
| DE | Differential Evolution | Single | [Storn and Price (1997)](https://link.springer.com/article/10.1023/a:1008202821328) |
| ABC |Artificial Bee Colony | Single | [Karaboga (2005)](https://abc.erciyes.edu.tr/pub/tr06_2005.pdf) |
| ASMO | Adaptive Surrogate Modelling based Optimization | Single, Surrogate | [Wang et al.(2014)](https://www.sciencedirect.com/science/article/pii/S1364815214001698) |
| EGO | Efficient Global Optimization | Single, Surrogate | [Jones (1998)](https://link.springer.com/article/10.1023/A:1008306431147)|

<br>

**Algorithms for multi-objective optimization:**

| Abbreviation | Full Name |   Optimization Label   |   References  |
|--------------|-----------| ----------|---------------|
| MOEA/D | Multi-objective Evolutionary Algorithm based on Decomposition | Multiple | [Zhang, Li (2007)](https://ieeexplore.ieee.org/document/4358754)|
| NSGA-II| Nondominated Sorting Genetic Algorithm II | Multiple | [Deb et al. (2002)](https://ieeexplore.ieee.org/document/996017)|
| NSGA-III| Nondominated Sorting Genetic Algorithm III| Multiple | [Deb, Jain (2014)](https://ieeexplore.ieee.org/document/6600851)|
| RVEA | Reference Vector guided Evolutionary Algorithm | Multiple | [Cheng et al. (2016)](https://ieeexplore.ieee.org/document/7386636)|
|MO-ASMO|Multi-Objective Adaptive Surrogate Modelling-based Optimization| Multiple, Surrogate | [Gong et al. (2015)](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015WR018230)|

💡 **Noted:** The label `Surrogate` indicates solving computationally expensive optimization problem. Its introduction can be referred to Surrogate Model of this Tutorial.

<br>

All algorithms above inherit from the `algorithmABC` class from `UQPyL.optimization`, which contain the fixed function, `run`. And, optimization history and results are saved into `Result` class.

<br>

API Reference:

**Class `algorithmABC`:**

```
The algorithmABC is the base class of algorithms from UQPyL.optimization, which contain the fixed function, `run`.
```

**Constructor**

```
__init__

- Description:

Initializes a new instance of the algorithm.

Note: The hyper-parameters are categorized into two parts: a. fixed parameters for all methods; b. special parameters.

- Fixed Parameters for all methods:
    - maxFEs(int) : The algorithm terminates when the number of objective function evaluations reaches `maxFEs`. 

    - maxIterTimes(int) :  The algorithm stops when the number of iterations reaches `maxIterTimes`.

    - maxTolerateTimes(int) : A threshold that defines how many consecutive times a certain condition can be tolerated before stopping the algorithm. Default, None, means unavailable. 

    - tolerate(float) : If the improvement in the objective value is smaller than this value, it counts toward termination based on maxTolerateTimes.

    - verboseFlag(bool): If set to `True`, the method will output detailed logs or progress messages during execution. Useful for debugging or tracking the optimization process. Default: True.

    - saveFlag(bool): If set to `True`, the optimization result would be saved to `./Result/Data`. Default: False.

    - logFlag(bool): If set to `True`, the verbose result would be simultaneously saved to `./Result/Log`. Default: False.

Other hyper-parameters vary between different optimization algorithms. Please refer to each specific method for details.

For example, the Genetic Algorithm(GA) extra accepts `proC`, `proM`, `disC`, `disM` to control optimization behavior.
```

**Methods**

```
run

- Description:

Execute the optimization

- Parameters:
    - problem(Problem) : The problem defined. This must be specified.

- Returns:
    - `Result` : A Class that containing optimization history and results.
```

**Class `Result`:**

It is used to store and manage the history and results of optimization.

**Attributes**

```
- bestDecs(np.2darray) : The best decision variables found during optimization. 

- bestTrueDecs(np.2darray) : If exist variables set 'discrete' or 'int', the 'bestDecs' need transform to 'bestTrueDecs'. 

- bestObjs(np.2darray) : The objective function values corresponding to the best solutions found.

- bestTrueObjs(np.2darray) : If the optType set to 'max', the 'bestObjs' need transform to 'bestTrueObjs'.

- bestCons(np.2darray) : The constraint values for the best solutions found. 

- bestMetric(float) : The best metric value for multi-objective (e.g., HV, LGD).

- bestFeasible(bool) : Indicates whether the best solution found satisfies all constraints.

- appearFEs(int) : The number of function evaluations when the best solution was found.

- appearIters(int) : The iteration number when the best solution was found.

- historyBestDecs(dict) : A dictionary recording the best decision variables found at different function evaluation (FE) counts. Each key is an FE number, and the corresponding value is the best decision vector up to that point.

- historyBestTrueDecs(dict) : Similar to `historyBestDecs`, but stores the transformed decision variables considering discrete or integer types.

- historyBestObjs(dict) : A record of the best objective function values found at various FE counts. 

- historyBestTrueObjs(dict) : Stores the best true objective values, transformed if optType is 'max', at each FE point.

- historyBestCons(dict) : Constraint values of the best solutions found at each recorded FE. 

- historyBestMetrics(dict) : The performance metric (e.g., HV, IGD) of the best solution found at different FE steps.

- historyDecs(dict) : All decision variables evaluated at specific FEs. Useful for analysis or visualization of the decision space over time.

- historyObjs(dict) : Objective values record at each FE count. 

- historyCons(dict) : Constraint values record at each FE count.
```

## 🍄‍🟫 How to optimization

**For single-objective optimization:**

```python
# Step 1: from `UQPyL.problems` import Sphere as example:

from UQPyL.problems.single_objective import Sphere

# Create an instance of Sphere with 15 dimensions, [-100, 100]^n 
problem = Sphere(nInput = 15, ub = 100, lb = -100)

# Step 2: from `UQPyL.optimization` import optimization algorithm

# Use GA as example
from UQPyL.optimization.single_objective import GA

ga = GA(
        nPop = 50, # The size of evolutionary population.
        maxFEs = 50000, # The maximum of function evaluation.
        verboseFlag = True, # The flag to enable verbose.
        saveFlag = True, # The flag to save history and results.
        logFlag = True, # The flag to save verbose.
        )

res = ga.run(problem = problem) # `res` is the instance of Result class

print( res.bestDecs ) # If exist int or discrete, use `print(res.bestTrueDecs)`
print( res.bestObjs ) # But advise: `print(res.bestTrueObjs)`

```

**For multi-objective:**

```python
# Step 1: from `UQPyL.problems` import ZDT1 as example:
from UQPyL.problem.multi_objective import ZDT1

# Use default setting
problem = ZDT1()

# Step 2: from `UQPyL.optimization` import optimization algorithm

# Use NSGA-II as example:

from UQPyL.optimization import NSGAII

nsgaii = NSGAII(
            nPop = 50, # The size of evolutionary population.
            maxFEs = 20000, # The maximum of function evaluation.
            verboseFlag = True, # The flag to enable verbose.
            saveFlag = True, # The flag to save history and results.
            logFlag = True, # The flag to save verbose.
            )

res = nsgaii.run(problem = problem)

print( res.bestDecs ) 
print( res.bestObjs ) # The shape is (N, 2)

```

## 🍒 Check the result of optimization(.hdf)

When creating an instance of an optimization algorithm, setting the `saveFlag` or `logFlag` to `True` will trigger automatic saving of the results to specific folders — either 'Result/Data' or 'Result/Log'.

The output file containing the analysis results is named following the format `A_B_C_D_I.hdf`, where: where: A denotes the name of the optimization algorithm used; B represents the name of the problem being analyzed; C denotes the dimension of the problem; D denotes the name of objective; I is the index indicating the iteration or repetition of the same analysis.

For example, a file named 'GA_Sphere_D15_M1_1.hdf' refers to the first execution of the Genetic Algorithm on the Sphere problem with 15 dimensions.

To visualize the results, we recommend installing H5Web, a VSCode extension that allows interactive viewing of HDF5 files directly within the editor.

<figure align="center">
  <img src="./pic/H5Web.png" alt="H5Web in VSCode" width="500"/>
  <figcaption>H5Web in VSCode</figcaption>
</figure>
