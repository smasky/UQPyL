import importlib

import numpy as np
import pytest

from UQPyL.optimization.soea import GA
from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.runtime import OptReader
from UQPyL.optimization.runtime.storage import SqliteStorage
from UQPyL.problem import Problem


def makeProblem(tmpPath, multi=False):
    def objective(X):
        if multi:
            return np.column_stack((X.sum(axis=1), ((X - 0.5) ** 2).sum(axis=1)))
        return X.sum(axis=1, keepdims=True)

    problem = Problem(nInput=2, nObj=2 if multi else 1, nCon=1,
                      lb=[10, 20], ub=[20, 30], objFunc=objective,
                      conFunc=lambda X: X[:, :1] - 17,
                      optType=['max', 'min'] if multi else 'max')
    problem.workDir = str(tmpPath)
    return problem


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('frequency', [1, 3, 10, None])
def test_sparse_history_preserves_curves_final_results_and_real_values(tmp_path, frequency, multi):
    problem = makeProblem(tmp_path, multi)
    methodClass = NSGAII if multi else GA
    options = dict(nPop=8, maxFEs=72, verboseFlag=False, logFlag=False, saveFlag=False)
    reference = methodClass(historyFreq=1, **options).run(problem, seed=12)
    method = methodClass(historyFreq=frequency, **options)
    result = method.run(problem, seed=12)
    np.testing.assert_array_equal(result.bestDecs, reference.bestDecs)
    np.testing.assert_array_equal(result.bestObjs, reference.bestObjs)
    np.testing.assert_array_equal(result.bestCons, reference.bestCons)
    assert result.bestMetric == reference.bestMetric
    history = result.history
    for name in ['metrics', 'iterToFEs', 'bestObjHistory', 'bestMetricHistory', 'numBestHistory', 'improvedHistory']:
        assert getattr(history, name) == getattr(reference.history, name)
    allKeys = history.iterToFEs
    expected = [key for i, key in enumerate(allKeys)
                if frequency is not None and (i == 0 or key[0] % frequency == 0)]
    if allKeys[-1] not in expected:
        expected.append(allKeys[-1])
    assert history.snapshotIterToFEs == expected
    assert len(history.populations) == len(history.bests) == len(expected)
    for key, population, best in zip(expected, history.populations, history.bests):
        index = reference.history.snapshotIterToFEs.index(key)
        for name in ['decs', 'objs', 'cons']:
            np.testing.assert_array_equal(population[name], reference.history.populations[index][name])
        np.testing.assert_array_equal(best['bestObjs'], reference.history.bests[index]['bestObjs'])
    assert result.toDict()['history']['snapshot_iter_to_fes'] == expected
    assert result.extra['history_freq'] == frequency
    np.testing.assert_array_equal(method.saveResult()['iterToFEs'], allKeys)
    # A subsequent run must not mutate the first result or its sparse snapshots.
    saved = history.populations[-1]['decs'].copy()
    method.run(problem, seed=13)
    np.testing.assert_array_equal(history.populations[-1]['decs'], saved)


def test_sqlite_frequency_is_independent_and_does_not_copy_history(tmp_path, monkeypatch):
    problem = makeProblem(tmp_path)
    method = GA(nPop=4, maxFEs=36, saveFlag=True, saveFreq=2,
                historyFreq=None, verboseFlag=False, logFlag=False)
    saveSnapshot = SqliteStorage.saveSnapshot
    observed = []
    paths = []

    def trackSave(self, session, obj, result, isFinal=False):
        paths.append(session.db_path)
        observed.append((result.iters, isFinal, len(result.history.populations), len(result.history.iterToFEs)))
        return saveSnapshot(self, session, obj, result, isFinal=isFinal)

    monkeypatch.setattr(SqliteStorage, 'saveSnapshot', trackSave)
    result = method.run(problem, seed=15)
    assert len(result.history.populations) == 1
    for iteration, isFinal, populationCount, statCount in observed:
        if not isFinal:
            assert iteration % 2 == 0
            assert populationCount == statCount == 0
    assert observed[-1][1] is True and observed[-1][2] == 1
    reader = OptReader(paths[-1])
    try:
        assert len(reader.list_snapshots()) == len(observed)
        assert reader.get_run_summary()['status'] == 'finished'
        np.testing.assert_array_equal(reader.load_last_best().objs, result.bestObjs)
    finally:
        reader.close()


@pytest.mark.parametrize('frequency', [1, 10, None])
def test_short_run_has_one_final_snapshot_without_duplicate(tmp_path, frequency):
    result = GA(nPop=4, maxFEs=4, historyFreq=frequency, saveFlag=False,
                verboseFlag=False).run(makeProblem(tmp_path), seed=1)
    assert len(result.history.populations) == 1
    assert result.history.snapshotIterToFEs == result.history.iterToFEs


@pytest.mark.parametrize('frequency', [0, -1, True, 1.5, '10'])
def test_history_frequency_validation(frequency):
    with pytest.raises(ValueError, match='historyFreq'):
        GA(historyFreq=frequency)
    method = GA()
    with pytest.raises(ValueError, match='historyFreq'):
        method.set('historyFreq', frequency)
    assert method.historyFreq == 10


@pytest.mark.parametrize('moduleName,className', [
    ('soea.ga', 'GA'), ('soea.de', 'DE'), ('soea.pso', 'PSO'),
    ('soea.abc', 'ABC'), ('soea.csa', 'CSA'), ('soea.sce_ua', 'SCE_UA'),
    ('soea.ml_sce_ua', 'ML_SCE_UA'), ('moea.nsga_ii', 'NSGAII'),
    ('moea.nsga_iii', 'NSGAIII'), ('moea.moea_d', 'MOEAD'),
    ('moea.rvea', 'RVEA'), ('expensive.ego', 'EGO'),
    ('expensive.asmo', 'ASMO'), ('expensive.moasmo', 'MOASMO'),
])
def test_all_optimizers_forward_history_configuration(moduleName, className):
    methodClass = getattr(importlib.import_module('UQPyL.optimization.' + moduleName), className)
    method = methodClass(historyFreq=None)
    assert method.historyFreq is None and method.get('historyFreq') is None
    method.set('historyFreq', 7)
    assert method.historyFreq == method.get('historyFreq') == 7
