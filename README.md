# UQPyL: Uncertainty Quantification Python Lab

<p align="center"><img src="./docs_v2/assets/logo.png" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) [![CI](https://github.com/smasky/UQPyL/actions/workflows/ci.yml/badge.svg)](https://github.com/smasky/UQPyL/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/smasky/UQPyL/branch/dev/graph/badge.svg)](https://codecov.io/gh/smasky/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

[English](README.md) | [中文](README_CN.md) | [Documentation](https://uqpyl.readthedocs.io)

UQPyL is a Python library for uncertainty quantification, optimization, inference, calibration, and surrogate modeling.
It defines problems once and reuses them across UQ workflows.

## What UQPyL solves

UQPyL provides a shared problem interface for common uncertainty quantification workflows.

Define the model or decision problem once, then reuse it for:

- design of experiments
- sensitivity and uncertainty analysis
- optimization
- Bayesian-style inference
- model calibration
- surrogate modeling

After a problem is defined, all UQPyL modules can work with the same object and remain connected across workflows.

<p align="center">
  <img src="./docs_v2/assets/architecture.png" alt="UQPyL architecture overview" width="1000"/>
</p>

## Problem abstraction

`Problem` is the main entry point of UQPyL. It turns a model, benchmark function, or decision task into a reusable object that other modules can use.

To build a `Problem`, define:

| Part | Role |
|---|---|
| Input space | Number of variables, bounds, labels, and variable types. |
| Evaluation rule | How input samples become objectives and optional constraints. |
| Additional information | Optimization direction, problem name, and metadata for algorithms, outputs, logs, and saved results. |

For example:

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    # Input space
    nInput=2, lb=-1.0, ub=1.0,

    # Evaluation rule
    nObj=1, objFunc=objFunc,

    # Additional information
    optType="min", name="Sphere2D",
)
```

This is enough for workflows that only need evaluated objectives or constraints, including DOE, analysis, optimization, inference, and surrogate modeling.

For many hydrological simulation problems, the workflow needs more than final objective values. Calibration and uncertainty analysis may need the simulated and observed time series, and the valid observation mask to remain available throughout the method.

`ModelProblem` extends `Problem` for this case. It adds:

| Extra part | Role |
|---|---|
| `simFunc` | Runs the model and returns simulated series or fields. |
| `obs` / `mask` | Stores observed data and marks valid entries for simulation-observation comparison. |

Use `Problem` by default. Use `ModelProblem` when a method needs simulation-process semantics, such as calibration methods that compare `sim` with `obs`. See the [documentation](https://uqpyl.readthedocs.io) for detailed usage.

<p align="center">
  <img src="./docs_v2/assets/Problem.webp" alt="Problem and ModelProblem comparison" width="1000"/>
</p>

For hydrological applications, the hard part is often not the UQ algorithm itself, but connecting an external model, preparing inputs, running simulations, and collecting outputs. [hydroPilot](https://github.com/smasky/hydroPilot) is designed for that model-operation layer. It can be used with UQPyL when you want hydroPilot to manage hydrological model runs and UQPyL to handle sampling, analysis, calibration, optimization, inference, or surrogate modeling.

## Architecture overview

UQPyL is organized around one shared `problem` abstraction and a set of functional modules built around it.

| Type | Module | Purpose |
|---|---|---|
| Core | `problem` | Define parameter spaces, evaluation rules, objectives, constraints, simulations, and related metadata. |
| Function | `doe` | Generate design samples for experiments, analysis, initialization, and modeling. |
| Function | `analysis` | Analyze how input variables affect model or objective outputs. |
| Function | `optimization` | Search for single-objective, multi-objective, or expensive-model optima. |
| Function | `inference` | Run MCMC-style parameter inference. |
| Function | `calibration` | Calibrate simulation models against observations. |
| Function | `surrogate` | Train and evaluate surrogate models for expensive evaluations. |
| Support | `runtime`, `viz` | Save structured run results, logs, intermediate states, and provide visualization utilities. |

## Quick start

### Optimization with `Problem`

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

algorithm = SCE_UA(maxFEs=200)

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
    # Here we assume each parameter sample returns one simulated series
    # with 3 time steps, so the output shape is (n_samples, 3, 1).
    sim = np.zeros((X.shape[0], 3, 1))
    sim[:, 0, 0] = X[:, 0] * 1.0
    sim[:, 1, 0] = X[:, 0] * 2.0
    sim[:, 2, 0] = X[:, 0] * 3.0
    return sim


problem = ModelProblem(
    nInput=1, nObj=1,
    lb=0.0, ub=2.0,
    simFunc=simFunc, obs=obs,
    name="LinearScaleModel",
)

X = np.linspace(0.5, 1.5, 32).reshape(-1, 1)
result = GLUE(metric="rmse").run(problem, X, threshold=0.2)
```

## Modules at a glance

The table lists representative methods, not the full API. See the [documentation](https://uqpyl.readthedocs.io) for complete module-specific usage and API details.

| Module | Representative methods |
|---|---|
| `doe` | `LHS`, `FFD`, `Random`, `Sobol`, `SaltelliDesign`, `FASTDesign`, `MorrisDesign` |
| `analysis` | `Sobol`, `FAST`, `RBDFAST`, `Morris`, `RSA`, `DeltaTest`, `MARS` |
| `optimization` | `GA`, `PSO`, `DE`, `SCE_UA`, `NSGAII`, `NSGAIII`, `MOEAD`, `RVEA`, `EGO` |
| `inference` | `MH`, `AMH`, `MH_Gibbs`, `DEMC`, `DREAM_ZS` |
| `calibration` | `GLUE`, `SUFI2`, `ES`, `IES` |
| `surrogate` | `RBF`, `GPR`, `KRG`, `LinearRegression`, `PolynomialRegression`, `AutoTuner` |

## Runtime output and saving

Most runnable methods expose three common runtime controls.

| Option | Role |
|---|---|
| `verboseFlag` | Print progress and summary in the terminal. |
| `logFlag` | Write more complete runtime logs when supported. |
| `saveFlag` | Save structured runtime results, usually sqlite. |

```python
algorithm = SCE_UA(maxFEs=200, verboseFlag=True, logFlag=True, saveFlag=True)
result = algorithm.run(problem, seed=123)
```

Example terminal output:

```text
Algorithm: SCE-UA
Problem: Sphere2D
nInput: 2
nObj: 1
maxFEs: 200
maxIters: 1000
SCE-UA | iter=10 eval=84 best=4.3210e-03 cv=0 time=0.0s
SCE-UA | iter=20 eval=154 best=2.1500e-04 cv=0 time=0.0s
Optimization finished
  algorithm        : SCE-UA
  status           : finished
  iterations       : 27
  evaluations      : 203
  best value       : 1.0000e-04
  best X           : [1.0000e-02, -0.0000e+00]
  constraint viol. : 0
  elapsed          : 0.0s
```

With `saveFlag=True`, a run can produce saved artifacts for later reading by module-specific readers.

| Save option | Rule |
|---|---|
| `saveFlag=True` | Enable structured result saving for the current run. |
| `saveFreq` | For optimization methods, save intermediate snapshots every `saveFreq` iterations and always save the final result at the end. |
| Reader access | Saved sqlite results can be loaded later with module-specific readers such as `OptReader`, `AnaReader`, or `CalReader`. |

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

For UQPyL 2.0, please cite the preprint:

Wu, M., Sun, R., Xu, P., Yang, X., Hu, P., & Duan, Q. UQPyL 2.0: An Open-Source Python Package for Uncertainty Quantification and Optimization. Available at SSRN: https://ssrn.com/abstract=5393295 or http://dx.doi.org/10.2139/ssrn.5393295

```bibtex
@misc{wu2025uqpyl2,
  title = {UQPyL 2.0: An Open-Source Python Package for Uncertainty Quantification and Optimization},
  author = {Wu, Mengtian and Sun, Ruochen and Xu, Pengcheng and Yang, Xu and Hu, Pengjie and Duan, Qingyun},
  year = {2025},
  note = {SSRN preprint},
  doi = {10.2139/ssrn.5393295},
  url = {https://ssrn.com/abstract=5393295}
}
```

For UQPyL 1.0, please cite:

Wang, C., Duan, Q., Tong, C. H., Di, Z., & Gong, W. (2016). A GUI platform for uncertainty quantification of complex dynamical models. *Environmental Modelling & Software*, 76, 1-12. https://doi.org/10.1016/j.envsoft.2015.11.004

```bibtex
@article{wang2016uqpyl,
  title = {A GUI platform for uncertainty quantification of complex dynamical models},
  author = {Wang, Chen and Duan, Qingyun and Tong, Charles H. and Di, Zhenhua and Gong, Wei},
  journal = {Environmental Modelling & Software},
  volume = {76},
  pages = {1--12},
  year = {2016},
  doi = {10.1016/j.envsoft.2015.11.004}
}
```

## Contributing

Contributions are welcome. Useful areas include new algorithms, model interfaces, benchmark problems, examples, tests, and documentation improvements.

## License

UQPyL is released under the MIT License. See [LICENSE.md](LICENSE.md).
