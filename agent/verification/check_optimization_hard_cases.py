"""Higher-dimensional, narrow-feasible and disconnected-front accuracy audit."""
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.spatial.distance import cdist

from UQPyL.optimization.soea import GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.optimization.expensive import EGO, ASMO, MOASMO
from UQPyL.problem import Problem

quiet = dict(verboseFlag=False, logFlag=False, saveFlag=False, historyFreq=None)
seeds = [5, 17, 41]
singleClasses = [GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA]
section = sys.argv[1]
output = Path(__file__).with_name(f'1002-optimization-hard-{section}.json')
records = []


def save(row):
    records.append(row)
    output.write_text(json.dumps(records, indent=2))
    if len(records) % 10 == 0:
        print(section, len(records), 'records', flush=True)


def makeSingle(cls, budget):
    options = dict(ngs=3) if cls in [SCE_UA, ML_SCE_UA] else dict(nPop=32)
    return cls(maxFEs=budget, maxIters=1000, tolerate=None, **options, **quiet)


def execute(method, problem, formula, seen, seed, metadata, initial=None):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = method.run(problem, seed=seed, initialPop=initial)
    elapsed = time.perf_counter() - start
    allX = np.vstack(seen)
    assert len(allX) == result.FEs
    np.testing.assert_allclose(problem.unit_to_space(problem.space_to_unit(allX)), allX, rtol=1e-12, atol=1e-14)
    allY, allC = formula(allX)
    returnedY, returnedC = formula(result.bestDecs)
    np.testing.assert_allclose(result.bestObjs, returnedY, rtol=1e-12, atol=1e-14)
    if allC is None:
        violation = np.zeros(len(allX))
    else:
        np.testing.assert_allclose(result.bestCons, returnedC)
        violation = np.maximum(allC, 0).sum(axis=1)
    feasible = violation <= 0
    assert result.bestFeasible == bool(np.any(feasible))
    if problem.nObj == 1:
        if np.any(feasible):
            np.testing.assert_allclose(returnedY[0, 0], np.min(allY[feasible, 0]), rtol=1e-12, atol=1e-14)
        else:
            assert np.isclose(np.maximum(returnedC, 0).sum(), violation.min(), rtol=1e-12, atol=1e-14)
    elif np.any(feasible):
        validY = allY[feasible]
        for point in returnedY:
            assert not np.any(np.all(validY <= point, axis=1) & np.any(validY < point, axis=1))
        for point in validY:
            assert np.any(np.all(returnedY <= point, axis=1))
    else:
        assert len(result.bestDecs) == 0
        assert np.isclose(result.minViolation, violation.min())
    firstFeasible = np.flatnonzero(feasible)
    row = dict(**metadata, method=type(method).__name__, seed=seed, passed=True,
               evaluations=result.FEs, iterations=result.iters, elapsed_seconds=elapsed,
               found_feasible=bool(np.any(feasible)), min_violation=float(violation.min()),
               first_feasible_evaluation=int(firstFeasible[0] + 1) if len(firstFeasible) else None,
               warnings=[str(w.message) for w in caught])
    return result, allY, feasible, row


if section == 'dimension':
    for dimension in [10, 30]:
        rotation, _ = np.linalg.qr(np.random.default_rng(123 + dimension).normal(size=(dimension, dimension)))
        for scenario in ['rotated_ellipsoid', 'rotated_rastrigin']:
            def formula(x):
                z = (x - .123) @ rotation
                if scenario == 'rotated_ellipsoid':
                    y = np.sum(z*z * np.geomspace(1e-6, 1., dimension), axis=1)
                else:
                    y = 10 * dimension + np.sum(z*z - 10 * np.cos(2 * np.pi * z), axis=1)
                return y[:, None], None
            for cls in singleClasses:
                for seed in seeds:
                    previous = np.inf
                    for budget in [1000, 4000]:
                        seen = []
                        def objective(x):
                            seen.append(x.copy())
                            return formula(x)[0]
                        problem = Problem(nInput=dimension, nObj=1, lb=-5, ub=5, objFunc=objective)
                        method = makeSingle(cls, budget)
                        result, allY, _, row = execute(method, problem, formula, seen, seed,
                            dict(scenario=scenario, dimension=dimension, budget=budget, known_optimum=0.))
                        best = float(result.bestObjs[0, 0])
                        assert best >= -1e-10 and best <= previous + 1e-12
                        previous = best
                        initialCount = 3 * (2 * dimension + 1) if cls in [SCE_UA, ML_SCE_UA] else 32
                        initialBest = float(np.min(allY[:initialCount]))
                        row.update(best=best, initial_best=initialBest, fraction_remaining=best / initialBest)
                        save(row)

elif section == 'narrow':
    for radius in [.03, .15]:
        optimum = (1.2 - radius)**2
        def formula(x):
            return np.sum((x - .2)**2, axis=1, keepdims=True), np.sum((x - .8)**2, axis=1, keepdims=True) - radius**2
        for cls in [*singleClasses, EGO, ASMO]:
            for warm in ([False, True] if cls in [EGO, ASMO] else [False]):
                for seed in seeds:
                    seen = []
                    def objective(x):
                        seen.append(x.copy())
                        return formula(x)[0]
                    problem = Problem(nInput=4, nObj=1, nCon=1, lb=0, ub=1,
                                      objFunc=objective, conFunc=lambda x: formula(x)[1])
                    budget = 80 if cls in [EGO, ASMO] else 2500
                    if cls in [EGO, ASMO]:
                        inner = GA(nPop=20, maxFEs=160, maxIters=8, tolerate=None, **quiet)
                        method = cls(nInit=16, maxFEs=budget, maxIters=100, optimizer=inner, **quiet)
                    else:
                        method = makeSingle(cls, budget)
                    initial = np.full((1, 4), .8) if warm else None
                    result, _, feasible, row = execute(method, problem, formula, seen, seed,
                        dict(scenario='narrow_ball', radius=radius, budget=budget,
                             known_optimum=optimum, feasible_start=warm), initial)
                    row['best_feasible'] = float(result.bestObjs[0, 0]) if np.any(feasible) else None
                    row['optimality_gap'] = row['best_feasible'] - optimum if np.any(feasible) else None
                    if np.any(feasible):
                        assert row['optimality_gap'] >= -1e-10
                    save(row)
        for budget in [80, 2500]:
            for seed in seeds:
                x = np.random.default_rng(seed).random((budget, 4))
                y, c = formula(x)
                good = c[:, 0] <= 0
                save(dict(scenario='narrow_ball', radius=radius, budget=budget, feasible_start=False,
                          method='UniformRandom', seed=seed, passed=True, evaluations=budget,
                          found_feasible=bool(np.any(good)), min_violation=float(np.maximum(c, 0).min()),
                          best_feasible=float(y[good].min()) if np.any(good) else None,
                          warnings=[]))

elif section == 'front':
    for scenario in ['curved_front', 'disconnected_front']:
        disconnected = scenario == 'disconnected_front'
        t = np.linspace(0, 1, 1001) if not disconnected else np.r_[np.linspace(.1, .25, 501), np.linspace(.75, .9, 501)]
        reference = np.column_stack([t*t, (1-t)**2])
        def formula(x):
            penalty = np.sum((x[:, 1:] - .37)**2, axis=1)
            y = np.column_stack([x[:, 0]**2 + penalty, (1 - x[:, 0])**2 + penalty])
            c = None
            if disconnected:
                t = x[:, 0]
                c = np.minimum((t - .1) * (t - .25), (t - .75) * (t - .9))[:, None]
            return y, c
        for cls in [NSGAII, NSGAIII, MOEAD, RVEA, MOASMO]:
            for seed in seeds:
                seen = []
                def objective(x):
                    seen.append(x.copy())
                    return formula(x)[0]
                problem = Problem(nInput=6, nObj=2, nCon=int(disconnected), lb=0, ub=1,
                                  objFunc=objective, conFunc=(lambda x: formula(x)[1]) if disconnected else None)
                budget = 72 if cls is MOASMO else 600
                if cls is MOASMO:
                    inner = NSGAII(nPop=24, maxFEs=144, hvFlag=False, **quiet)
                    method = cls(nInit=24, pct=.25, maxFEs=budget, maxIters=100, optimizer=inner, hvFlag=False, **quiet)
                else:
                    method = cls(nPop=24, maxFEs=budget, maxIters=100, hvFlag=False, **quiet)
                result, _, _, row = execute(method, problem, formula, seen, seed,
                    dict(scenario=scenario, dimension=6, budget=budget))
                assert len(result.bestDecs) > 0
                distances = cdist(reference, result.bestObjs)
                # Independent SciPy computation; do not use the metric under test.
                row['igd_reference_grid'] = float(np.mean(np.min(distances, axis=1)))
                row['max_reference_gap'] = float(np.max(np.min(distances, axis=1)))
                row['front_size'] = len(result.bestDecs)
                row['t_min'] = float(result.bestDecs[:, 0].min())
                row['t_max'] = float(result.bestDecs[:, 0].max())
                row['both_segments'] = bool(np.any(result.bestDecs[:, 0] <= .25) and np.any(result.bestDecs[:, 0] >= .75))
                save(row)
elif section == 'matched':
    for radius in [.03, .15]:
        def formula(x):
            return np.sum((x - .2)**2, axis=1, keepdims=True), np.sum((x - .8)**2, axis=1, keepdims=True) - radius**2
        for cls in singleClasses:
            for seed in seeds:
                seen = []
                def objective(x):
                    seen.append(x.copy())
                    return formula(x)[0]
                problem = Problem(nInput=4, nObj=1, nCon=1, lb=0, ub=1,
                                  objFunc=objective, conFunc=lambda x: formula(x)[1])
                method = makeSingle(cls, 80)
                _, _, _, row = execute(method, problem, formula, seen, seed,
                    dict(scenario='matched_first_80', radius=radius, budget=80))
                prefix = np.vstack(seen)[:80]
                _, cons = formula(prefix)
                row['prefix_count'] = len(prefix)
                row['prefix_found_feasible'] = bool(np.any(cons[:, 0] <= 0))
                row['prefix_min_violation'] = float(np.maximum(cons, 0).min())
                save(row)
else:
    raise ValueError(section)

print('completed', section, len(records), 'records; warnings', sum(bool(row['warnings']) for row in records), flush=True)
