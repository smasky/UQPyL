import numpy as np
import pytest

from UQPyL.problem import Space


@pytest.mark.parametrize("types", [[0, 3], [0, -1], [0, .5], [0, np.nan],
                                    [0, np.inf], [0, 2**32], ["0", "1"],
                                    [[0, 1]], [0], 1, [False, True]])
def testInvalidVariableTypesAreRejected(types):
    with pytest.raises(ValueError, match="varType"):
        Space(2, ub=10, lb=0, varType=types)


def testValidMixedVariableTypesRoundTrip():
    space = Space(3, ub=10, lb=0, varType=[0., 1., 2.], varSet={2: [20, 40]})
    values = np.array([[2., 8., 40.]])
    np.testing.assert_array_equal(space.unit_to_space(space.space_to_unit(values)), values)


@pytest.mark.parametrize("methodName", ["Sobol", "FAST", "RBDFAST", "RSA", "DeltaTest", "Morris"])
def testAnalysisShapesAndSelectedLabels(methodName):
    import UQPyL.analysis as analysis
    from UQPyL.doe import SaltelliDesign, FASTDesign, MorrisDesign, LHS
    from UQPyL.problem import Problem
    problem = Problem(nInput=2, nObj=2, lb=0., ub=1., objLabels=['flow', 'temperature'],
                      objFunc=lambda X: X)
    design, count = {"Sobol": (SaltelliDesign(), 64), "FAST": (FASTDesign(), 129),
                     "Morris": (MorrisDesign(), 10)}.get(methodName, (LHS(), 80))
    X, meta = design.sampleWithMeta(problem, count, seed=1)
    method = getattr(analysis, methodName)(verboseFlag=False)
    options = dict(meta=meta) if methodName in ('Sobol', 'FAST', 'Morris') else {}
    selected = method.analyze(problem, X, Y=X, index=1, **options)
    single = method.analyze(problem, X, Y=X[:, 1], **options)
    for left, right in zip(selected.metrics, single.metrics):
        assert left.rowLabels == ['temperature']
        np.testing.assert_allclose(left.values, right.values)
    with pytest.raises(ValueError, match='row'):
        method.analyze(problem, X, Y=X[:-1], **options)


@pytest.mark.parametrize("bad", ['missingCons', 'rows', 'columns', 'valid'])
def testPreEvaluatedPopulationContract(bad):
    from UQPyL.problem import Problem
    from UQPyL.optimization import Population
    from UQPyL.optimization.soea import GA
    def unexpected(X):
        raise AssertionError('Pre-evaluated data must not be evaluated again')
    problem = Problem(nInput=1, nObj=1, nCon=1, lb=0., ub=1., objFunc=unexpected, conFunc=unexpected)
    pop = Population([[.2], [.8]], np.ones((1 if bad == 'rows' else 2, 2 if bad == 'columns' else 1)),
                     None if bad == 'missingCons' else np.zeros((2, 1)))
    method = GA(nPop=2, maxIters=0, verboseFlag=False, logFlag=False, saveFlag=False)
    if bad == 'valid':
        assert method.run(problem, initialPop=pop).FEs == 0
    else:
        with pytest.raises(ValueError):
            method.run(problem, initialPop=pop)


@pytest.mark.parametrize("name,count", [('ES', 2), ('IES', 4), ('SUFI2', 1)])
def testCalibrationReusesSimulationBatches(name, count):
    import UQPyL.calibration as calibration
    from UQPyL.problem import ModelProblem
    calls = []
    def simulate(X):
        values = np.column_stack([X[:, 0], X[:, 1] ** 2, X[:, 0] + X[:, 1]])[:, :, None]
        calls.append(values.copy())
        return values
    problem = ModelProblem(nInput=2, lb=0., ub=3., simFunc=simulate,
                           obs=np.array([[1.], [2.], [0.]]), mask=np.array([[False], [False], [True]]))
    method = getattr(calibration, name)(**({'maxIters': 3} if name == 'IES' else {}))
    result = method.run(problem, np.array([[.5, .8], [1., 1.5], [2., 2.]]),
                        **({'eliteSize': 2} if name == 'SUFI2' else {}))
    assert len(calls) == count
    if name != 'SUFI2':
        np.testing.assert_array_equal(result.posteriorSims, calls[-1][:, :, 0])
        np.testing.assert_allclose(result.diagnostics['scores'], method.score(result.posteriorSims))


def testSavedOptimizationResultPreservesMissingMetrics(tmp_path):
    from UQPyL.optimization.moea import NSGAII
    from UQPyL.optimization.runtime import OptReader
    from UQPyL.problem import Problem
    from UQPyL.viz.optimization import _history_xy
    problem = Problem(nInput=1, nObj=2, nCon=1, lb=0., ub=1.,
                      objFunc=lambda X: np.column_stack([X, 1-X]), conFunc=lambda X: np.ones_like(X))
    problem.workDir = str(tmp_path)
    method = NSGAII(nPop=4, maxIters=2, saveFlag=True, saveFreq=1, verboseFlag=False, logFlag=False)
    original = method.run(problem, seed=1)
    reader = OptReader(next(tmp_path.rglob('*.sqlite3')))
    try:
        result = reader.load_result()
        assert not result.bestFeasible
        assert len(result.history.populations) == len(result.history.snapshotIterToFEs) == 3
        np.testing.assert_array_equal(result.history.populations[-1]['decs'], reader.load_last_population().decs)
        assert result.toDict()['best_feasible'] is False
        np.testing.assert_array_equal(result.candidateObjs, original.candidateObjs)
        rows = reader.list_snapshots()
        for row, value in zip(rows, [None, 2., 3., 3.]):
            reader.conn.execute('UPDATE snapshot SET hypervolume=? WHERE snapshotId=?', (value, row['snapshotId']))
        result = reader.load_result()
        x, y = _history_xy(result, 'fe')
        np.testing.assert_array_equal(x, [rows[1]['fe'], rows[2]['fe']])
        np.testing.assert_array_equal(y, [2., 3.])
    finally:
        reader.close()


@pytest.mark.parametrize('verbose', [False, True])
def testOptimizationElapsedTimeIsSaved(tmp_path, verbose):
    import time
    from UQPyL.problem import Problem
    from UQPyL.optimization.soea import GA
    from UQPyL.optimization.runtime import OptReader
    def objective(X):
        time.sleep(.005)
        return np.sum(X**2, axis=1, keepdims=True)
    problem = Problem(nInput=2, nObj=1, lb=0., ub=1., objFunc=objective)
    problem.workDir = str(tmp_path)
    result = GA(nPop=4, maxIters=2, tolerate=None, verboseFlag=verbose,
                saveFlag=True, logFlag=False, saveFreq=1).run(problem, seed=1)
    assert result.runtime >= .015
    with OptReader(next(tmp_path.rglob('*.sqlite3'))) as reader:
        elapsed = [row['elapsed'] for row in reader.list_snapshots()]
        assert elapsed == sorted(elapsed) and elapsed[0] > 0
        assert elapsed[-1] == reader.get_run()['runtime'] == result.runtime


def testReadersRejectOtherDomains(tmp_path):
    from UQPyL.problem import Sphere
    from UQPyL.optimization.soea import GA
    from UQPyL.calibration import CalReader
    from UQPyL.inference import InfReader
    from UQPyL.analysis.runtime import AnaReader
    from UQPyL.optimization.runtime import OptReader
    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    GA(nPop=4, maxIters=0, saveFlag=True, logFlag=False, verboseFlag=False).run(problem, seed=1)
    path = next(tmp_path.rglob('*.sqlite3'))
    for cls in [CalReader, InfReader, AnaReader]:
        assert cls.list_runs(tmp_path) == []
        with pytest.raises(ValueError, match='database'):
            cls(path)
    assert len(OptReader.list_runs(tmp_path)) == 1


def testProblemStarImportExportsPublicNames():
    import UQPyL.problem as problem
    namespace = {}
    exec('from UQPyL.problem import *', namespace)
    assert all(isinstance(name, str) for name in problem.__all__)
    assert all(namespace[name] is getattr(problem, name) for name in problem.__all__)


@pytest.mark.parametrize('kernelName', ['RBF', 'Matern', 'RationalQuadratic', 'Constant', 'DotProduct'])
def testKernelDiagonalMatchesFullKernel(kernelName):
    from UQPyL.surrogate.gp import kernel as kernels
    from UQPyL.surrogate.gp.kernel.c_kernel_ import Constant
    from UQPyL.surrogate.gp.kernel.dot_kernel_ import DotProduct
    kernel = ({"Constant": Constant, "DotProduct": DotProduct}.get(kernelName) or getattr(kernels, kernelName))()
    X = np.random.default_rng(4).normal(size=(80, 3))
    np.testing.assert_allclose(kernel.diag(X), np.diag(kernel(X)), rtol=1e-14)


@pytest.mark.parametrize('family', ['GPR', 'KRG'])
def testMeanPredictionSkipsVarianceWork(family, monkeypatch):
    import importlib
    from UQPyL.surrogate.gp import GPR
    from UQPyL.surrogate.kriging import KRG
    model = {'GPR': GPR, 'KRG': KRG}[family]()
    X = np.linspace(0, 1, 12)[:, None]
    model.fitModel(X, np.sin(X))
    mean, variance = model.predict(X, returnVar=True)
    module = importlib.import_module(model.__class__.__module__)
    def unexpected(*args, **kwargs):
        raise AssertionError('Mean prediction must skip variance solves')
    monkeypatch.setattr(module, 'solve_triangular' if family == 'GPR' else 'lstsq', unexpected)
    np.testing.assert_allclose(model.predict(X), mean)


@pytest.mark.parametrize('isModel', [False, True])
@pytest.mark.parametrize('target', [None, 'objs', 'cons'])
def testEvaluationCallsOnlyRequestedCallback(isModel, target):
    from UQPyL.problem import Problem, ModelProblem
    calls = []
    def objective(X, *context):
        calls.append('objs')
        return X
    def constraint(X, *context):
        calls.append('cons')
        return X - .5
    options = dict(nInput=1, nObj=1, nCon=1, lb=0., ub=1., objFunc=objective, conFunc=constraint)
    problem = ModelProblem(simFunc=lambda X: X[:, :, None], **options) if isModel else Problem(**options)
    result = problem.evaluate([[.2]], target=target)
    assert calls == (['objs', 'cons'] if target is None else [target])
    assert (result.objs is not None) == (target in (None, 'objs'))
    assert (result.cons is not None) == (target in (None, 'cons'))


def testCalibrationStorageAndSummaryAvoidDuplicatePayloads(tmp_path, monkeypatch):
    from UQPyL.calibration import GLUE, CalReader
    from UQPyL.problem import ModelProblem
    problem = ModelProblem(nInput=1, lb=0., ub=1., obs=np.ones((30, 1)),
                           simFunc=lambda X: np.repeat(X[:, :, None], 30, axis=1))
    problem.workDir = str(tmp_path)
    result = GLUE(saveFlag=True).run(problem, np.linspace(0, 1, 100)[:, None], threshold=10.)
    with CalReader(next(tmp_path.rglob('*.sqlite3'))) as reader:
        sizes = dict(reader.conn.execute('SELECT name, length(payload) FROM artifact'))
        assert set(sizes) == {'result', 'summary'}
        assert sum(sizes.values()) < 1.1 * sizes['result']
        def unexpected(*args, **kwargs):
            raise AssertionError('Summary must not load full results')
        monkeypatch.setattr(reader, 'get_artifacts', unexpected)
        monkeypatch.setattr(reader, 'load_result', unexpected)
        assert reader.get_run_summary()['best_score'] == result.summary()['best_score']


@pytest.mark.parametrize('methodIndex', range(14))
def testEveryOptimizationConfigurationCanBeRestored(tmp_path, methodIndex):
    from test_optimization_stopping_semantics import METHODS, MULTI, makeMethod
    from UQPyL.problem import Problem
    from UQPyL.optimization.runtime import OptReader
    import warnings
    cls = METHODS[methodIndex]
    nObj = 2 if cls in MULTI else 1
    problem = Problem(nInput=2, nObj=nObj, lb=0., ub=1.,
                      objFunc=lambda X: np.column_stack([np.sum(X**2, axis=1)] * nObj))
    problem.workDir = str(tmp_path)
    original = makeMethod(cls, 0)
    original.saveFlag = True
    original.run(problem, seed=2)
    with OptReader(next(tmp_path.rglob('*.sqlite3'))) as reader:
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            restored = reader.load_algorithm()
        assert type(restored) is cls
        for name in ['maxFEs', 'maxIter', 'maxTolerates', 'tolerate', 'verboseFlag', 'saveFlag', 'historyFreq']:
            assert getattr(restored, name) == getattr(original, name)
        if hasattr(original, 'optimizer'):
            assert any('manually' in str(message.message) for message in messages)


def testRunIdCollisionDoesNotChangePreviousRun(tmp_path, monkeypatch):
    import sqlite3
    from UQPyL.core import runtime
    from UQPyL.optimization.soea import GA
    from UQPyL.problem import Sphere
    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    monkeypatch.setattr(runtime, 'make_run_id', lambda *args: 'fixed_run')
    method = GA(nPop=4, maxIters=0, saveFlag=True, logFlag=False, verboseFlag=False)
    method.run(problem, seed=1)
    with pytest.raises(sqlite3.IntegrityError):
        method.run(problem, seed=1)
    with sqlite3.connect(next(tmp_path.rglob('*.sqlite3'))) as conn:
        assert conn.execute('SELECT status FROM run').fetchone()[0] == 'finished'


def testLogOnlyRunsHaveDistinctIdsAndFiles(tmp_path):
    from UQPyL.optimization.soea import GA
    from UQPyL.problem import Sphere
    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    method = GA(nPop=4, maxIters=0, logFlag=True, saveFlag=False, verboseFlag=False)
    ids = []
    for _ in range(2):
        method.run(problem, seed=1)
        ids.append(method.runId)
    assert len(set(ids)) == 2
    assert {p.stem for p in tmp_path.rglob('*.log')} == set(ids)


def testOptimizationConfigurationRoundTrip(tmp_path):
    from UQPyL.optimization.soea import GA
    from UQPyL.optimization.runtime import OptReader
    from UQPyL.problem import Sphere
    problem = Sphere(nInput=2)
    problem.workDir = str(tmp_path)
    original = GA(nPop=4, maxFEs=18, maxIters=0, tolerate=None, maxTolerates=3,
                  verboseFlag=False, logFlag=False, saveFlag=True, saveFreq=2, historyFreq=3)
    original.run(problem, seed=1)
    reader = OptReader(next(tmp_path.rglob('*.sqlite3')))
    try:
        restored = reader.load_algorithm()
        for key in ['maxFEs', 'maxIter', 'tolerate', 'maxTolerates', 'verboseFlag',
                    'logFlag', 'saveFlag', 'saveFreq', 'historyFreq']:
            assert getattr(restored, key) == getattr(original, key)
        assert reader._parseValue('1+2') == '1+2'
    finally:
        reader.close()
