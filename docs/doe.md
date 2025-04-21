# Design of Experiment

---

## 🌳 What is the Design of Experiment

**Design of Experiment (DoE)** refers to a structured, strategic approach for selecting samples withing decision space to efficiently system's behavior. DoE aims to extract the the maximum amount of information with a minimal number of experiments or simulations. Therefore, DoE methods are the essential for uncertainty quantification, optimization, or surrogate modeling.

## 🌿 Overview of `UQPyL.DoE` module

The `DoE` module of UQPyL provides several design of experiments methods to meet various user requirements.

<br>

DoE methods include:

- **Full Factorial Designs (FFD):** FFD would explore all possible combinations of factors at different levels. 

- **Latin Hypercube Sampling (LHS):** LHS is designed to obtain uniform and efficient coverage of decision space with fewer samples. DoE module provides multiple variants of LHS, e.g., `classic`, `center`, `maximin`, `center_maximin`, `correlation`. 

- **Monte Carlo Sampling:** Random sampling is the most basic method. Each sample is selected independently from the specified probability distributions.

- **Sobol Sequence:** This method is a quasi-random low-discrepancy sequence designed to uniformly fill the parameter space more effectively than purely random methods. 

- **Saltelli Sequence:** It is a variant of Sobol Sequence, more suitable for variance-based global sensitivity analysis.

- **Morris Sequence:** This method is designed for Morris Method in sensibility analysis. 

All methods in `UQPyL.DoE` module inherits from `SamplerABC`.
Here, we give the API reference of `SamplerABC`:

<br>

**Class `SamplerABC`:**

```
The SamplerABC is the base class for DoE methods in UQPyL. It provides a unified sample function that allows users to generate samples efficiently. 
```

**Constructor:**
```
__init__

- Description:

Initializes a new instance of DoE method.

Note: The hyper-parameters vary between different DoE methods. Please refer to each specific method for details.

For example, the __init__ method of LHS accepts criterion('classic', 'center' ... 'correlated') and iterations (if required) as input arguments.

```

**Method:**

```
sample

- Description:

Generate samples from the decision space.

There are two modes of operation:

1. If no `problem` is provided:
   - The function uses `nt` (number of samples) and `nx` (number of variables)
     to generate samples in the normalized space [0, 1]^nx.

2. If a `problem` is provided:
   - The function uses `nt` and `problem.nInput` to generates samples. 
     And samples would be normalized to the variable bounds 
     defined by the `problem` instance.

- Parameters:
    - nt(int): The parameters would determine the sampling number of each method.
    - nx(int): The dimensions of sampling, optional. Default: None.
    - problem(Problem): The Problem instance defined, optional. Default: None.

- Returns:
    - np.2darray: A 2D Numpy Array of generated array.

    - The number of rows typically corresponds to `nt`, 
      such as in LHS and Random sampling methods.
    - For some methods like Sobol or Saltelli sequences, 
      the number of rows may differ due to internal padding requirements.
```

## 🍁 How to obtain samples

```python

# Step 1 : Create an instance of DoE method
from UQPyL.DoE import LHS

# Use classic LHS method
lhs = LHS(criterion = "classic") 

# Step 2: Generate a unit sample set with 100 samples in 10 dimensions
# the shape is (nt, nx)
X = lhs.sample(nt = 100, nx = 10)

# Optional: Provide a problem definition to scale the samples accordingly
from UQPyL.problems.single_objective import Sphere

problem = Sphere(nInput = 10)

# Samples are now generated within the bounds defined by the `problem`
# the shape is (nt, problem.nInput)
X = lhs.sample(nt = 100, problem = problem)

```