"""Independent optimization audit; findings are recorded, not suppressed."""
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from optimization_test_support import METHODS, MULTI, QUIET, makeMethod
from UQPyL.optimization.core import NDSort, crowdingDist
from UQPyL.optimization.metric import GD, IGD
from UQPyL.optimization.soea import ABC
from UQPyL.optimization.soea import SCE_UA, ML_SCE_UA
from UQPyL.optimization.moea import RVEA
from UQPyL.problem import Problem

records = []


def record(kind, **values):
    records.append(dict(kind=kind, **values))


def referenceRanks(objs, cons):
    cv = np.maximum(cons, 0).sum(axis=1)
    remaining = list(range(len(objs)))
    result = np.zeros(len(objs), dtype=int)
    rank = 0
    while remaining:
        rank += 1
        def dominates(a, b):
            if cv[a] != cv[b]:
                return cv[a] < cv[b]
            return cv[a] == 0 and np.all(objs[a] <= objs[b]) and np.any(objs[a] < objs[b])
        front = [b for b in remaining if not any(dominates(a, b) for a in remaining)]
        result[front] = rank
        remaining = [i for i in remaining if i not in front]
    return result


for seed in range(20):
    rng = np.random.default_rng(seed)
    for dimension in [2, 3, 5]:
        objs = rng.integers(-3, 4, (24, dimension)).astype(float)
        cons = rng.integers(-2, 3, (24, 2)).astype(float)
        expected = referenceRanks(objs, cons)
        actual, last = NDSort(objs, cons)
        assert np.array_equal(actual, expected)
        for count in [1, 7, 24]:
            partial, cutoff = NDSort(objs, cons, count)
            expectedCutoff = np.sort(expected)[count - 1]
            assert cutoff == expectedCutoff
            assert np.array_equal(partial <= cutoff, expected <= cutoff)
        record('nondominated_reference', seed=seed, dimension=dimension, passed=True)


for cls in METHODS:
    for seed in [7, 19, 31]:
        for direction in ['min', 'max']:
            sign = 1 if direction == 'min' else -1
            nObj = 2 if cls in MULTI else 1
            evaluated = []
            def objectives(x):
                evaluated.extend(x.copy().tolist())
                first = np.sum((x - .3) ** 2, axis=1)
                second = np.sum((x - .7) ** 2, axis=1)
                return sign * np.column_stack([first, second])[:, :nObj]
            problem = Problem(nInput=2, nObj=nObj, lb=0, ub=1,
                              objFunc=objectives, optType=direction)
            method = makeMethod(cls, 3)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = method.run(problem, seed=seed)
            x = result.bestDecs
            actual = result.bestObjs
            expected = sign * np.column_stack([np.sum((x - .3) ** 2, axis=1),
                                               np.sum((x - .7) ** 2, axis=1)])[:, :nObj]
            assert np.allclose(actual, expected, rtol=1e-12, atol=1e-14)
            assert np.all((x >= 0) & (x <= 1))
            assert result.FEs == len(evaluated)
            seen = np.array(evaluated)
            retained = True
            bestSeen = None
            if nObj == 1:
                bestSeen = float(np.min(np.sum((seen - .3) ** 2, axis=1)))
                retained = bool(np.isclose(float(actual[0, 0]) * sign, bestSeen, rtol=1e-12, atol=1e-14))
            else:
                seenObjs = np.column_stack([np.sum((seen - .3) ** 2, axis=1),
                                           np.sum((seen - .7) ** 2, axis=1)])
                for row in actual * sign:
                    assert not np.any(np.all(seenObjs <= row, axis=1) & np.any(seenObjs < row, axis=1))
            record('public_run', method=cls.__name__, seed=seed, direction=direction,
                   evaluations=result.FEs, iterations=result.iters, passed=retained,
                   best_seen=bestSeen, returned=actual.tolist(),
                   warnings=[str(w.message) for w in caught])


for cls, seed in [(SCE_UA, 19), (ML_SCE_UA, 19)]:
    method = makeMethod(cls, 3)
    originalCce = method._cce
    trace = []
    def traceCce(simplex, *args):
        before = simplex.objs[:, 0].copy()
        candidate = originalCce(simplex, *args)
        trace.append(dict(simplex_scores=before.tolist(),
                          assumed_worst=float(before[-1]), actual_worst=float(np.max(before)),
                          candidate=float(candidate.objs[0, 0])))
        return candidate
    method._cce = traceCce
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1,
                      objFunc=lambda x: np.sum((x - .3) ** 2, axis=1, keepdims=True))
    method.run(problem, seed=seed)
    invalid = [row for row in trace if row['assumed_worst'] < row['actual_worst']]
    record('sce_simplex_order', method=cls.__name__, seed=seed, calls=len(trace),
           wrong_worst_count=len(invalid), examples=invalid[:3], passed=not invalid)


# A successful employed-bee move must clear its consecutive failure history.
for seed in range(10):
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    method = ABC(nPop=4, **QUIET)
    method.setup(problem, seed)
    pop = method.initPop(4, initialPop=np.array([[.6], [.9], [.8], [.7]]))
    roles = np.array([1, 0, 0, 0])
    counts = np.array([6., 0, 0, 0])
    before = float(pop.objs[0, 0])
    pop, counts = method.updateEmployedBees(pop, roles, counts)
    improved = float(pop.objs[0, 0]) < before
    nextCounts = counts.copy()
    nextCounts[0] += 1
    abandoned = bool(method.checkLimitTimes(roles.copy(), nextCounts, 6)[0] == 2)
    record('abc_counter', seed=seed, before=before, after=float(pop.objs[0, 0]),
           improved=improved, failure_count=float(counts[0]), abandoned_after_next_failure=abandoned,
           passed=not improved or counts[0] == 0)


for scale in [1., 1e-200, 1e200]:
    for metric in [GD, IGD]:
        actual = metric([[scale, scale]], [[0., 0.]])
        expected = math.hypot(scale, scale)
        record('distance_scale', metric=metric.__name__, scale=scale,
               actual=str(actual), expected=expected,
               passed=math.isclose(actual, expected, rel_tol=1e-14, abs_tol=0))


front = np.array([[0., 4.], [1., 3.], [2., 2.], [3., 1.], [4., 0.]])
vectors = np.array([[1., 0.], [.5, .5], [0., 1.]])
selector = RVEA(**QUIET)
baseline = selector.environmentSelection(front, vectors, .5)
for scale in [1., 1e-200, 1e200]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        selected = selector.environmentSelection(front * scale, vectors, .5)
    record('rvea_common_scale', scale=scale, baseline=baseline.tolist(), actual=selected.tolist(),
           passed=np.array_equal(selected, baseline), warnings=[str(w.message) for w in caught])

front = np.array([[-1., 1.], [-.5, .5], [0., 0.], [.5, -.5], [1., -1.]])
baseline = crowdingDist(front)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    actual = crowdingDist(front * 1e308)
record('crowding_scale', baseline=baseline.tolist(), actual=actual.tolist(),
       passed=np.array_equal(actual, baseline), warnings=[str(w.message) for w in caught])

suffix = '-after' if '--after' in sys.argv else ''
output = Path(__file__).with_name(f'1002-optimization-review{suffix}.json')
output.write_text(json.dumps(records, indent=2, default=lambda value: value.item()))
print('records', len(records), 'failed controls', sum(not r['passed'] for r in records))
for row in records:
    if not row['passed']:
        print(row)
