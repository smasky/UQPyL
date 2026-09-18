# Design of Experiment API

## `UQPyL.doe`

The `doe` module generates design samples from a `Problem` input space. Samplers first generate samples in unit space, then map them to the problem bounds through `problem.unit_to_space()`.

### Import

```python
from UQPyL.doe import LHS, Random, FFD, Sobol
```

### Public Objects

| Object | Role |
|---|---|
| `Sampler` | Base class for DOE samplers. |
| `Random` | Uniform random sampling. |
| `LHS` | Latin hypercube sampling. |
| `FFD` | Full factorial design. |
| `Sobol` | Sobol low-discrepancy sequence sampling. |
| `SaltelliDesign` | Saltelli design for Sobol sensitivity analysis. |
| `FASTDesign` | FAST design for sensitivity analysis. |
| `MorrisDesign` | Morris trajectory design for sensitivity analysis. |

## `Sampler`

`Sampler` is the base interface used by most DOE classes.

### Methods

| Method | Returns | Meaning |
|---|---|---|
| `sample(problem, nSamples=None, seed=None, nt=None, *, output="real")` | `np.ndarray` | Generate samples in the problem space. |
| `sampleWithMeta(problem, nSamples=None, seed=None, nt=None, *, output="real")` | `(np.ndarray, dict)` | Generate samples and metadata. |

| Parameter | Meaning |
|---|---|
| `problem` | A `ProblemBase` instance, such as `Problem`, `ModelProblem`, or a built-in benchmark direct problem. |
| `nSamples` | Number of samples or base sample size, depending on the sampler. |
| `seed` | Optional random seed. |
| `nt` | Legacy alias for `nSamples`. If both are provided, they must match. |

Example:

```python
from UQPyL.doe import LHS
from UQPyL.problem import Sphere


problem = Sphere(nInput=3)
sampler = LHS()

X = sampler.sample(problem, nSamples=10, seed=123)
print(X.shape)
```

Use `sampleWithMeta()` when downstream methods need design metadata:

```python
X, meta = sampler.sampleWithMeta(problem, nSamples=10, seed=123)

print(meta["designType"])
```

## `Random`

Uniform random sampler.

```python
Random()
```

| Method | Meaning |
|---|---|
| `sample(problem, nSamples=None, seed=None, nt=None, *, output="real")` | Generate random samples in the problem space. |
| `sampleWithMeta(problem, nSamples=None, seed=None, nt=None, *, output="real")` | Generate random samples with metadata. |

Metadata:

| Key | Meaning |
|---|---|
| `designType` | Always `"random"`. |
| `seed` | Seed used to initialize the random generator. |

## `LHS`

Latin hypercube sampler.

```python
LHS(criterion="classic", iterations=5)
```

| Parameter | Meaning |
|---|---|
| `criterion` | LHS criterion. One of `"classic"`, `"center"`, `"maximin"`, `"center_maximin"`, or `"correlation"`. |
| `iterations` | Number of candidate designs tested by optimized criteria. |

`criterion="correlation"` selects the candidate with the smallest maximum absolute
Pearson correlation between distinct input columns in unit space. Perfect correlations
of +1 or -1 are included. `iterations` must be a positive integer. With one input
variable or one sample, it returns the first classic LHS design because there is no
correlation criterion to optimize. It emits no candidate-search messages. Discrete
decoding may change correlations in the returned real values.

| Method | Meaning |
|---|---|
| `sample(problem, nSamples=None, seed=None, nt=None, *, output="real")` | Generate LHS samples in the problem space. |
| `sampleWithMeta(problem, nSamples=None, seed=None, nt=None, *, output="real")` | Generate LHS samples with metadata. |

Metadata:

| Key | Meaning |
|---|---|
| `designType` | Always `"lhs"`. |
| `criterion` | Criterion used by the sampler. |
| `iterations` | Iteration count for optimized criteria. |
| `seed` | Seed used to initialize the random generator. |

Example:

```python
from UQPyL.doe import LHS
from UQPyL.problem import Sphere


problem = Sphere(nInput=2)
sampler = LHS(criterion="maximin", iterations=10)

X, meta = sampler.sampleWithMeta(problem, nSamples=20, seed=123)
print(X.shape)
print(meta["criterion"])
```

## `FFD`

Full factorial design.

```python
FFD()
```

Unlike most samplers, `FFD` uses `levels` instead of `nSamples`.

| Method | Returns | Meaning |
|---|---|---|
| `sample(problem, levels, seed=None)` | `np.ndarray` | Generate the full factorial grid in the problem space. |
| `sampleWithMeta(problem, levels, seed=None)` | `(np.ndarray, dict)` | Generate the grid with metadata. |

| Parameter | Meaning |
|---|---|
| `levels` | Scalar level count applied to all variables, or one level count per variable. |
| `seed` | Accepted for API consistency. It does not affect the deterministic grid. |

Metadata:

| Key | Meaning |
|---|---|
| `designType` | Always `"full_fact"`. |
| `levels` | Level count used for each variable. |
| `seed` | Always `None`. |

Example:

```python
from UQPyL.doe import FFD
from UQPyL.problem import Sphere


problem = Sphere(nInput=2)
sampler = FFD()

X, meta = sampler.sampleWithMeta(problem, levels=[3, 4])
print(X.shape)
print(meta["levels"])
```

## `Sobol`

Sobol low-discrepancy sequence sampler.

```python
Sobol(scramble=True, skipValue=0)
```

| Parameter | Meaning |
|---|---|
| `scramble` | Whether to scramble the Sobol sequence. |
| `skipValue` | Number of initial Sobol points to skip. |

| Method | Meaning |
|---|---|
| `sample(problem, nSamples=None, seed=None, nt=None, *, output="real")` | Generate Sobol samples in the problem space. |
| `sampleWithMeta(problem, nSamples, seed=None)` | Generate Sobol samples with metadata. |

Metadata:

| Key | Meaning |
|---|---|
| `designType` | Always `"sobol_sequence"`. |
| `scramble` | Whether scrambling was enabled. |
| `skipValue` | Number of skipped initial points. |
| `seed` | Seed used when `scramble=True`; otherwise `None`. |

`nSamples` should usually be a power of 2 for balanced Sobol samples. `skipValue` must be a non-negative integer and cannot exceed `nSamples`.

## `SaltelliDesign`

Saltelli design for Sobol sensitivity analysis.

```python
SaltelliDesign(scramble=True, skipValue=0, secondOrder=False)
```

| Parameter | Meaning |
|---|---|
| `scramble` | Whether to scramble the Sobol base sequence. |
| `skipValue` | Number of initial Sobol points to skip. |
| `secondOrder` | Whether to generate the second-order design. |

| Method | Meaning |
|---|---|
| `sample(problem, nSamples=None, seed=None, nt=None, *, output="real")` | Generate the design in the problem space. |
| `sampleWithMeta(problem, N, seed=None)` | Generate the design with metadata. |

For a problem with `D` inputs and base sample size `N`, output shape is:

| `secondOrder` | Shape |
|---|---|
| `False` | `((D + 2) * N, D)` |
| `True` | `((2 * D + 2) * N, D)` |

Metadata:

| Key | Meaning |
|---|---|
| `designType` | Always `"saltelli"`. |
| `N` | Base sample size. |
| `secondOrder` | Whether second-order design was generated. |
| `skipValue` | Number of skipped Sobol points. |
| `scramble` | Whether scrambling was enabled. |
| `blockSize` | Number of rows per base sample. |
| `seed` | Seed used when `scramble=True`; otherwise `None`. |

`N` should usually be a power of 2. `skipValue` must be a non-negative integer and cannot exceed `N`.

## `FASTDesign`

FAST design for sensitivity analysis.

```python
FASTDesign(M=4)
```

| Parameter | Meaning |
|---|---|
| `M` | FAST interference parameter. Must be a positive integer. |

| Method | Meaning |
|---|---|
| `sample(problem, nSamples=None, seed=None, nt=None, *, output="real")` | Generate the FAST design in the problem space. |
| `sampleWithMeta(problem, N, seed=None)` | Generate the FAST design with metadata. |

For a problem with `D` inputs and base sample size `N`, output shape is:

```text
(N * D, D)
```

`N` must satisfy:

```text
N > 4 * M^2
```

Metadata:

| Key | Meaning |
|---|---|
| `designType` | Always `"fast"`. |
| `N` | Base sample size. |
| `M` | FAST interference parameter. |
| `blockSize` | Rows per input variable. |
| `seed` | Seed used to initialize the random generator. |

## `MorrisDesign`

Morris trajectory design for sensitivity analysis.

```python
MorrisDesign(numLevels=4)
```

| Parameter | Meaning |
|---|---|
| `numLevels` | Number of Morris levels. Must be an even integer greater than or equal to 4. |

| Method | Meaning |
|---|---|
| `sample(problem, nSamples=None, seed=None, nt=None, *, output="real")` | Generate Morris trajectories in the problem space. |
| `sampleWithMeta(problem, numTrajectory, seed=None)` | Generate Morris trajectories with metadata. |

For a problem with `D` inputs and `numTrajectory` trajectories, output shape is:

```text
(numTrajectory * (D + 1), D)
```

Metadata:

| Key | Meaning |
|---|---|
| `designType` | Always `"morris"`. |
| `numTrajectory` | Number of trajectories. |
| `numLevels` | Number of Morris levels. |
| `trajectorySize` | Rows per trajectory, equal to `D + 1`. |
| `seed` | Seed used to initialize the random generator. |

Example:

```python
from UQPyL.doe import MorrisDesign
from UQPyL.problem import Sphere


problem = Sphere(nInput=3)
sampler = MorrisDesign(numLevels=4)

X, meta = sampler.sampleWithMeta(problem, numTrajectory=10, seed=21)
print(X.shape)
print(meta["trajectorySize"])
```


## Select the output space

All built-in samplers accept keyword-only `output="real"` (default) or `output="unit"` in `sample()` and `sampleWithMeta()`. Real samples can be evaluated directly. Unit output returns the generated `[0,1]` samples before decoding. With matching configuration, size and seed, both modes share the same unit design. Metadata records the selected `output`.

```python
U = LHS().sample(problem, 20, seed=123, output="unit")
X = LHS().sample(problem, 20, seed=123)  # equals problem.unit_to_space(U)
```

Integer/discrete values are decoded using equal-width bins. Discrete real values come from `varSet` and need not lie within that column's legacy encoding bounds. Standalone DOE and analysis calls still default to real samples.
