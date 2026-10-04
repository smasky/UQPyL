"""Independent public-run invariants and modest accuracy controls after repairs."""
import json
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from optimization_test_support import METHODS, MULTI, QUIET, makeMethod
from UQPyL.optimization.soea import GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA
from UQPyL.problem import Problem

records = []


def checkRun(method, problem, evaluateExpected, seen, seed, label, direction):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = method.run(problem, seed=seed)
    allX = np.vstack(seen)
    assert len(allX) == result.FEs
    # Round-trip validates integer/discrete membership as well as bounds.
    np.testing.assert_allclose(problem.unit_to_space(problem.space_to_unit(allX)), allX)
    expectedY, expectedC = evaluateExpected(allX)
    actualY, actualC = evaluateExpected(result.bestDecs)
    np.testing.assert_allclose(result.bestObjs, actualY * direction, rtol=1e-12, atol=1e-14)
    if expectedC is not None:
        np.testing.assert_allclose(result.bestCons, actualC)
        weights = np.ones(expectedC.shape[1]) if problem.conWgt is None else problem.conWgt.reshape(-1)
        cv = np.maximum(expectedC * weights, 0).sum(axis=1)
    else:
        cv = np.zeros(len(allX))
    feasible = cv <= 0
    assert result.bestFeasible == bool(np.any(feasible))
    if problem.nObj == 1:
        if np.any(feasible):
            assert np.isclose(actualY[0, 0], expectedY[feasible, 0].min(), rtol=1e-12, atol=1e-14)
        else:
            returnedCv = np.maximum(actualC * weights, 0).sum()
            assert np.isclose(returnedCv, cv.min(), rtol=1e-12, atol=1e-14)
    else:
        if np.any(feasible):
            valid = expectedY[feasible]
            for point in actualY:
                assert not np.any(np.all(valid <= point, axis=1) & np.any(valid < point, axis=1))
            # Every evaluated feasible point is weakly dominated by an archived
            # point: a missing non-dominated point cannot pass this control.
            for point in valid:
                assert np.any(np.all(actualY <= point, axis=1))
        else:
            assert len(result.bestDecs) == 0
            assert np.isclose(result.minViolation, cv.min())
    records.append(dict(kind='public_invariant', method=type(method).__name__, scenario=label, seed=seed,
                        evaluations=result.FEs, iterations=result.iters, passed=True,
                        warnings=[str(w.message) for w in caught]))


for cls in METHODS:
    for seed in [2, 11, 23]:
        method = makeMethod(cls, 4)
        nObj = 2 if cls in MULTI else 1
        direction = np.array([1, -1])[:nObj] if seed == 11 else np.full(nObj, -1 if seed == 23 else 1)
        for scenario in ['continuous', 'constrained', 'mixed', 'infeasible']:
            seen = []
            mixed = scenario == 'mixed'
            constrained = scenario != 'continuous'
            def evaluateExpected(x):
                unit = x.copy()
                if mixed:
                    unit[:, 1] /= 4
                    unit[:, 2] = (unit[:, 2] + 2) / 9
                first = np.sum((unit - .25)**2, axis=1)
                second = np.sum((unit - .75)**2, axis=1)
                y = np.column_stack([first, second])[:, :nObj]
                c = None
                if constrained:
                    c = np.column_stack([unit[:, 0] - .6, .1 - unit[:, 1]])
                    if scenario == 'infeasible':
                        c[:, 0] = unit[:, 0] + 1
                return y, c
            def objective(x):
                seen.append(x.copy())
                return evaluateExpected(x)[0] * direction
            options = dict(varType=[0, 1, 2], varSet={2: [-2., .5, 7.]}) if mixed else {}
            problem = Problem(nInput=3, nObj=nObj, nCon=2 if constrained else 0,
                              lb=[0, 0, 0], ub=[1, 4, 1] if mixed else [1, 1, 1],
                              objFunc=objective, optType=['min' if value == 1 else 'max' for value in direction],
                              conFunc=(lambda x: evaluateExpected(x)[1]) if constrained else None,
                              conWgt=[2., .5] if constrained else None, **options)
            checkRun(method, problem, evaluateExpected, seen, seed, scenario, direction)


for cls in [GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA]:
    for seed in [2, 11, 23]:
        for scenario in ['sphere', 'rosenbrock', 'rastrigin']:
            seen = []
            def objective(x):
                if scenario == 'sphere':
                    y = np.sum(x*x, axis=1)
                elif scenario == 'rosenbrock':
                    y = 100 * (x[:, 1] - x[:, 0]**2)**2 + (1 - x[:, 0])**2
                else:
                    y = 20 + np.sum(x*x - 10 * np.cos(2 * np.pi * x), axis=1)
                seen.extend(y.tolist())
                return y[:, None]
            problem = Problem(nInput=2, nObj=1, lb=-2, ub=2, objFunc=objective)
            options = dict(ngs=3) if cls in [SCE_UA, ML_SCE_UA] else dict(nPop=24)
            method = cls(maxFEs=1500, maxIters=100, tolerate=None, **options, **QUIET)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = method.run(problem, seed=seed)
            best = float(result.bestObjs[0, 0])
            assert best == min(seen)
            assert best >= -1e-12
            initialCount = 15 if cls in [SCE_UA, ML_SCE_UA] else 24
            records.append(dict(kind='accuracy', method=cls.__name__, scenario=scenario, seed=seed,
                                initial_best=min(seen[:initialCount]), best=best, known_optimum=0.,
                                evaluations=result.FEs, passed=True,
                                warnings=[str(w.message) for w in caught]))

output = Path(__file__).with_name('1002-optimization-extended.json')
output.write_text(json.dumps(records, indent=2))
print('records', len(records), 'all invariant checks passed')
for cls in [GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA]:
    print(cls.__name__, {scenario: float(np.median([row['best'] for row in records
                           if row['kind'] == 'accuracy' and row['method'] == cls.__name__ and row['scenario'] == scenario]))
                       for scenario in ['sphere', 'rosenbrock', 'rastrigin']})
print('warnings', [(row['method'], row['scenario'], row['seed'], row['warnings'])
                   for row in records if row['warnings']])
