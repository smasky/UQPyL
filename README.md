# UQPyL: Uncertainty Quantification Python Lab

<p align="center"><img src="./docs_v2/assets/logo.png" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) [![CI](https://github.com/smasky/UQPyL/actions/workflows/ci.yml/badge.svg)](https://github.com/smasky/UQPyL/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/smasky/UQPyL/branch/dev/graph/badge.svg)](https://codecov.io/gh/smasky/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

[English](README.md) | [中文](README_CN.md)

UQPyL is a Python library for uncertainty quantification, optimization, inference, calibration, and surrogate modeling.
It defines problems once and reuses them across UQ workflows.

## What UQPyL solves

UQPyL targets uncertainty problems in computational modeling: how uncertain parameters affect outputs, which inputs matter most, how to calibrate models against observations, and how to search for robust or optimal decisions.

These workflows are common across model-based domains, especially in hydrology, water resources, and water engineering.

| Task | Examples | What UQPyL provides |
|---|---|---|
| Explore parameter spaces | Generate candidate hydrological parameter sets. | Design of experiment methods for reproducible sampling. |
| Understand input influence | Identify which parameters dominate flow, load, or other model outputs. | Sensitivity and uncertainty analysis methods. |
| Search for good parameters | Calibrate parameters or optimize engineering decisions. | Single-objective, multi-objective, and expensive-model optimization algorithms. |
| Estimate plausible parameters | Sample parameter distributions under uncertainty. | MCMC-style inference methods. |
| Calibrate simulation models | Compare simulated series with observations. | Calibration methods based on `ModelProblem`. |
| Reduce expensive evaluations | Build a cheaper approximation of a slow model run. | Surrogate models and surrogate-assisted workflows. |

## Documentation

- Documentation: <https://uqpyl.readthedocs.io>
- Source code: <https://github.com/smasky/UQPyL>

## Core idea: define once, reuse everywhere

UQPyL does not own your model logic. Instead, you wrap your model or decision problem as a shared `problem` definition with:

| Part | Meaning |
|---|---|
| Input space | Variables, bounds, labels, and variable types. |
| Evaluation rule | How a batch of inputs becomes objectives, constraints, or simulations. |
| Optimization direction | Whether each objective is minimized or maximized. |
| Runtime identity | A name and metadata used by saved runs and summaries. |

Once defined, the same object can be reused by DOE, analysis, optimization, inference, surrogate workflows, and calibration. Some workflows also need explicit model-aware information such as simulations, observations, or masks.

## Problem abstraction

The `problem` module is the conceptual entry point of UQPyL.

| Abstraction | Role |
|---|---|
| `Problem` | For methods that only need final objective or constraint values. |
| `ModelProblem` | For methods that need explicit model-process semantics such as `sim`, `obs`, or masks. |

Both abstractions share the same foundation:

| Building block | Role |
|---|---|
| `Space` | Defines variables, bounds, labels, and variable types. |
| `Eval` | Standard return object for evaluated results. |

The module also includes benchmark problems such as `Sphere`, `Ackley`, `ZDT`, and `DTLZ`.

Use `Problem` when methods only need final objectives or constraints from candidate inputs. `Problem` can still be used for model-based problems when those final values are enough. Use `ModelProblem` only when methods need explicit simulations, observations, masks, or simulated-versus-observed comparison. In the current design, this mainly applies to calibration workflows.

<p align="center">
  <img src="./docs_v2/assets/Problem.webp" alt="Problem and ModelProblem comparison" width="1000"/>
</p>

For hydrological models, the hard part is often model connection rather than the algorithm itself. For that layer, we recommend [hydroPilot](https://github.com/smasky/hydroPilot).

## Architecture overview

UQPyL is organized around one shared `problem` abstraction and a set of functional modules built around it.

<p align="center">
  <img src="./docs_v2/assets/architecture.png" alt="UQPyL architecture overview" width="1000"/>
</p>

The figure summarizes the main idea: represent a modeling task as a shared `problem` definition, then reuse that definition across DOE, analysis, optimization, inference, calibration, and surrogate workflows, with unified outputs and optional runtime storage.

| Type | Module | Purpose |
|---|---|---|
| Core | `problem` | Define parameter spaces, evaluation rules, objectives, constraints, simulations, and related metadata. |
| Function | `doe` | Generate design samples for experiments, analysis, initialization, and modeling. |
| Function | `analysis` | Analyze how input variables affect model or objective outputs. |
| Function | `optimization` | Search for single-objective, multi-objective, or expensive-model optima. |
| Function | `inference` | Run MCMC-style parameter inference. |
| Function | `calibration` | Calibrate simulation models against observations. |
| Function | `surrogate` | Train and evaluate surrogate models for expensive evaluations. |

Visualization, runtime storage, logs, and readers are exposed through the functional modules rather than treated as primary entry points.

## Typical workflows

Common workflow patterns all lead to structured outputs and optional runtime storage. Workflows that only consume final objectives or constraints can use `Problem`, while workflows that need explicit `sim`, `obs`, or related comparison semantics use `ModelProblem`.

```text
Problem -> DOE -> Analysis -> outputs
Problem -> Optimization -> outputs
Problem -> Inference -> outputs
ModelProblem -> Calibration -> outputs
Problem -> DOE -> Surrogate -> Optimization
```

## Quick start examples

### Direct evaluation with `Problem`

```python
import numpy as np

from UQPyL.problem import Problem
from UQPyL.optimization.soea import SCE_UA


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    nInput=2, nObj=1,
    ub=1.0, lb=-1.0,
    objFunc=objFunc, optType="min",
    name="Sphere2D",
)

algorithm = SCE_UA(maxFEs=200, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
```

### Calibration with `ModelProblem`

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem

obs = np.array([[1.0], [2.0], [3.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    return X[:, :1][:, None, :] * obs[None, :, :]


problem = ModelProblem(
    nInput=1, nObj=1,
    lb=0.0, ub=2.0,
    simFunc=simFunc, obs=obs,
    name="LinearScaleModel",
)

X = np.linspace(0.5, 1.5, 32).reshape(-1, 1)
result = GLUE(metric="rmse", verboseFlag=False).run(problem, X, threshold=0.2)
```

Methods return structured result objects such as `OptResult`, `AnaResult`, `InfResult`, or `CalResult`.

## Modules at a glance

| Goal | Start with |
|---|---|
| Sample a parameter space | `doe` |
| Identify influential inputs | `analysis` |
| Search for good parameters | `optimization` with `SCE_UA` |
| Fit model parameters to observations | `calibration` with `ModelProblem` |
| Build a fast approximation of an expensive model | `surrogate` |

| Module | Representative methods |
|---|---|
| `doe` | `LHS`, `FFD`, `Random`, `Sobol`, `SaltelliDesign`, `FASTDesign`, `MorrisDesign` |
| `analysis` | `Sobol`, `FAST`, `RBDFAST`, `Morris`, `RSA`, `DeltaTest`, `MARS` |
| `optimization` | `GA`, `PSO`, `DE`, `SCE_UA`, `NSGAII`, `NSGAIII`, `MOEAD`, `RVEA`, `EGO` |
| `inference` | `MH`, `AMH`, `MH_Gibbs`, `DEMC`, `DREAM_ZS` |
| `calibration` | `GLUE`, `SUFI2`, `ES`, `IES` |
| `surrogate` | `RBF`, `GPR`, `KRG`, `LinearRegression`, `PolynomialRegression`, `AutoTuner` |

For many single-objective hydrological and engineering calibration problems, `SCE_UA` is a good starting point.

## Runtime output

Most runnable methods expose three common runtime controls.

| Option | Role |
|---|---|
| `verboseFlag` | Print progress and summary in the terminal. |
| `logFlag` | Write more complete runtime logs when supported. |
| `saveFlag` | Save structured runtime results, usually sqlite. |

With `saveFlag=True`, a run can produce saved artifacts for later reading by module-specific readers.

## More examples

Sensitivity analysis with `Sobol`:

```python
import numpy as np

from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    y = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1])**2 + 0.1 * X[:, 2]**4 * np.sin(X[:, 0])
    return y[:, None]


problem = Problem(
    nInput=3, nObj=1,
    lb=-np.pi, ub=np.pi,
    objFunc=objFunc, name="Ishigami",
)

X, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 512)
Y = problem.evaluate(X, target="objs").objs
result = Sobol(verboseFlag=False).analyze(problem, X, Y, meta=meta, target="objs")
```

## Installation

UQPyL requires Python 3.8 or newer.

```bash
pip install -U UQPyL
```

With plotting utilities:

```bash
pip install -U "UQPyL[viz]"
```

From source:

```bash
git clone https://github.com/smasky/UQPyL.git
cd UQPyL
pip install .
```

## Citation

Citation information for UQPyL 2.x will be updated. For UQPyL 1.0, see:

- <https://www.sciencedirect.com/science/article/pii/S1364815215300955>

## Contributing

Contributions are welcome. Useful areas include new algorithms, model interfaces, benchmark problems, examples, tests, and documentation improvements.

## License

UQPyL is released under the MIT License. See [LICENSE.md](LICENSE.md).
