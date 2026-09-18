import numpy as np
import pytest

from UQPyL.problem import Problem
from UQPyL.doe import LHS, Random, FFD, Sobol, SaltelliDesign, MorrisDesign, FASTDesign
from UQPyL.optimization import Population
from UQPyL.optimization.soea import GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.optimization.expensive import EGO, ASMO, MOASMO


QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def mixedProblem(nObj=1, optType="min"):
    def objectives(X):
        first = ((X[:, 0]-137)/100)**2 + (X[:, 1]-3)**2 + (X[:, 2]-20)**2/100
        second = ((X[:, 0]-183)/100)**2 + (X[:, 1]-5)**2 + (X[:, 2]-70)**2/100
        return np.column_stack([first, second])[:, :nObj]
    problem = Problem(nInput=3, nObj=nObj, lb=[100, 2, -7], ub=[200, 5, 8],
                      varType=[0, 1, 2], varSet={2: [10, 20, 70]}, objFunc=objectives, optType=optType)
    return problem


def test_mixed_encode_decode_endpoints_midpoints_and_copies():
    problem = mixedProblem()
    unit = np.array([[0., 0., 0.], [.37, .4, .5], [1., 1., 1.]])
    original = unit.copy()
    real = problem.unit_to_space(unit)
    np.testing.assert_allclose(real, [[100, 2, 10], [137, 3, 20], [200, 5, 70]])
    encoded = problem.space_to_unit(real)
    np.testing.assert_allclose(encoded, [[0, .125, 1/6], [.37, .375, .5], [1, .875, 5/6]])
    np.testing.assert_allclose(problem.unit_to_space(encoded), real)
    np.testing.assert_array_equal(unit, original)
    np.testing.assert_allclose(problem.canonicalize_unit(unit), encoded)
    np.testing.assert_allclose(problem.canonicalize_unit(encoded), encoded)


def test_fixed_and_fractional_integer_bounds_and_invalid_inputs():
    problem = Problem(nInput=2, nObj=1, lb=[7, 2.2], ub=[7, 4.8], varType=[0, 1], objFunc=lambda X: X[:, :1])
    real = problem.unit_to_space([[0, 0], [1, 1]])
    np.testing.assert_allclose(real, [[7, 3], [7, 4]])
    np.testing.assert_allclose(problem.space_to_unit(real), [[.5, .25], [.5, .75]])
    with pytest.raises(ValueError, match='legal integer'):
        problem.space_to_unit([[7, 3.2]])
    with pytest.raises(ValueError, match='outside'):
        problem.space_to_unit([[8, 3]])
    with pytest.raises(ValueError, match='Unit coordinates'):
        problem.unit_to_space([[0, 1.1]])
    with pytest.raises(ValueError, match='varSet'):
        mixedProblem().space_to_unit([[137, 3, 40]])


@pytest.mark.parametrize('sampler,count', [(LHS(), 8), (Random(), 8), (FFD(), [2, 3, 2]),
    (Sobol(), 8), (SaltelliDesign(), 8), (MorrisDesign(), 4), (FASTDesign(), 129)])
def test_all_sampler_output_modes_share_samples(sampler, count):
    problem = mixedProblem()
    unit, meta = sampler.sampleWithMeta(problem, count, seed=31, output='unit')
    real = sampler.sample(problem, count, seed=31)
    assert meta['output'] == 'unit'
    assert np.all((unit >= 0) & (unit <= 1))
    np.testing.assert_array_equal(real, problem.unit_to_space(unit))
    with pytest.raises(ValueError, match='output'):
        sampler.sample(problem, count, seed=31, output='wrong')


SO_FACTORIES = [lambda cls=cls: cls(nPop=12, maxFEs=36, maxIters=2, **QUIET)
                for cls in [GA, DE, PSO, ABC, CSA]]
SO_FACTORIES += [lambda cls=cls: cls(ngs=2, maxFEs=40, maxIters=1, **QUIET)
                 for cls in [SCE_UA, ML_SCE_UA]]
MO_FACTORIES = [lambda cls=cls: cls(nPop=12, maxFEs=36, maxIters=2, **QUIET)
                for cls in [NSGAII, NSGAIII, MOEAD, RVEA]]


@pytest.mark.parametrize('factory,nObj', [(f, 1) for f in SO_FACTORIES]+[(f, 2) for f in MO_FACTORIES])
def test_all_search_algorithms_keep_unit_population_and_export_real_solutions(factory, nObj):
    problem = mixedProblem(nObj)
    bounds = (problem.lb.copy(), problem.ub.copy())
    algorithm = factory()
    originalEvaluate = algorithm.evaluate
    seen = []
    def evaluate(pop):
        assert np.all((pop.decs >= 0) & (pop.decs <= 1))
        encoded = pop.decs.copy()
        originalEvaluate(pop)
        np.testing.assert_array_equal(pop.decs, encoded)
        real = problem.unit_to_space(encoded)
        np.testing.assert_allclose(pop.objs, problem.evaluate(real).objs)
        seen.extend(real.tolist())
        return pop
    algorithm.evaluate = evaluate
    result = algorithm.run(problem, seed=42)
    assert seen
    np.testing.assert_allclose(result.bestObjs, problem.evaluate(result.bestDecs).objs)
    for snapshot in result.history.populations:
        np.testing.assert_allclose(snapshot['objs'], problem.evaluate(snapshot['decs']).objs)
        assert set(snapshot['decs'][:, 2]).issubset({10, 20, 70})
    np.testing.assert_array_equal(problem.lb, bounds[0])
    np.testing.assert_array_equal(problem.ub, bounds[1])


def test_initial_real_population_encoding_and_result_history_isolation():
    problem = mixedProblem()
    initial = np.array([[120, 2, 10], [137, 3, 20], [180, 5, 70]])
    algorithm = GA(nPop=3, maxFEs=3, **QUIET)
    first = algorithm.run(problem, initialPop=initial, seed=1)
    np.testing.assert_array_equal(first.history.populations[0]['decs'], initial)
    np.testing.assert_array_equal(first.bestDecs, [[137, 3, 20]])
    algorithm.run(problem, seed=2)
    np.testing.assert_array_equal(first.history.populations[0]['decs'], initial)


def test_pre_evaluated_real_initial_population_is_encoded_without_evaluation():
    problem = mixedProblem()
    real = np.array([[137, 3, 20]])
    initial = Population(real, problem.evaluate(real).objs)
    algorithm = GA(nPop=1, maxFEs=1, **QUIET)
    algorithm.setup(problem, seed=1)
    pop = algorithm.initPop(1, initialPop=initial)
    assert algorithm.FEs == 0
    np.testing.assert_allclose(pop.decs, problem.space_to_unit(real))
    np.testing.assert_array_equal(initial.decs, real)


class RecordingSurrogate:
    def __init__(self, problem, model):
        self.problem = problem
        self.model = model
        self.training = []
        self.predictions = []

    def _check(self, X):
        assert np.all((X >= 0) & (X <= 1))
        np.testing.assert_allclose(X, self.problem.canonicalize_unit(X), atol=1e-15)

    def fit(self, X, Y):
        self._check(X)
        assert len(np.unique(X, axis=0)) == len(X)
        self.training.append((X.copy(), Y.copy()))
        self.model.fit(X, Y)
        return self

    def predict(self, X, **kwargs):
        self._check(X)
        self.predictions.append(X.copy())
        return self.model.predict(X, **kwargs)


@pytest.mark.parametrize('kind', ['EGO', 'ASMO', 'MOASMO'])
def test_real_surrogate_training_prediction_and_nested_search_use_canonical_unit(kind):
    from UQPyL.surrogate import MultiSurrogate
    from UQPyL.surrogate.kriging import KRG
    nObj = 2 if kind == 'MOASMO' else 1
    problem = mixedProblem(nObj)
    models = [KRG() for _ in range(nObj)]
    for model in models:
        model.rng = np.random.default_rng(12)
    model = models[0] if nObj == 1 else MultiSurrogate(nObj, models_list=models)
    recorder = RecordingSurrogate(problem, model)
    if kind == 'EGO':
        algorithm = EGO(nInit=8, maxFEs=9, maxIters=1, **QUIET)
        algorithm.surrogate = recorder
        algorithm.optimizer = GA(nPop=6, maxFEs=12, maxIters=1, **QUIET)
    elif kind == 'ASMO':
        algorithm = ASMO(nInit=8, maxFEs=9, maxIters=1, surrogate=recorder,
                         optimizer=GA(nPop=6, maxFEs=12, maxIters=1, **QUIET), **QUIET)
    else:
        algorithm = MOASMO(nInit=8, maxFEs=9, maxIters=1, pct=.25, surrogates=recorder,
                           optimizer=NSGAII(nPop=8, maxFEs=16, maxIters=1, **QUIET), **QUIET)
    result = algorithm.run(problem, seed=10)
    assert recorder.training and recorder.predictions
    for X, Y in recorder.training:
        np.testing.assert_allclose(Y, problem.evaluate(problem.unit_to_space(X)).objs)
    assert result.FEs > 8
    if kind == 'MOASMO':
        assert result.FEs == 10
    np.testing.assert_allclose(result.bestObjs, problem.evaluate(result.bestDecs).objs)
    rows = result.history.populations[-1]['decs']
    assert len(np.unique(rows, axis=0)) == len(rows)


def test_surrogate_training_merges_equivalent_discrete_encodings():
    algorithm = EGO(nInit=2, maxFEs=3, **QUIET)
    problem = mixedProblem()
    algorithm.setup(problem, 1)
    pop = Population([[.3, .3, .4], [.3, .45, .6]])
    algorithm.evaluate(pop)
    class Capture:
        def fit(self, X, Y):
            self.X, self.Y = X.copy(), Y.copy()
    model = Capture()
    algorithm._fitSurrogate(model, pop)
    assert model.X.shape == (1, 3)
    np.testing.assert_allclose(model.X, [[.3, .375, .5]])
    np.testing.assert_allclose(model.Y, pop.objs[:1])


def test_finite_discrete_search_stops_without_repeated_true_evaluations():
    problem = Problem(nInput=1, nObj=1, lb=-100, ub=100, varType=[2], varSet={0: [10, 20]},
                      objFunc=lambda X: X**2)
    algorithm = EGO(nInit=2, maxFEs=10, maxIters=4, **QUIET)
    algorithm.optimizer = GA(nPop=4, maxFEs=8, maxIters=1, **QUIET)
    result = algorithm.run(problem, initialPop=[[10], [20]], seed=3)
    assert result.FEs == 2
    np.testing.assert_array_equal(result.bestDecs, [[10]])


@pytest.mark.parametrize("optType", ["min", "max"])
def test_sqlite_npz_and_history_export_real_decisions(tmp_path, optType):
    from UQPyL.optimization import OptReader
    problem = mixedProblem(optType=optType)
    problem.workDir = str(tmp_path)
    algorithm = GA(nPop=6, maxFEs=12, maxIters=1, verboseFlag=False, logFlag=False,
                   saveFlag=True, saveFreq=1)
    result = algorithm.run(problem, seed=41)
    dbFiles = list(tmp_path.rglob('*.sqlite3'))
    assert len(dbFiles) == 1
    reader = OptReader(dbFiles[0])
    try:
        for snapshot in reader.list_snapshots():
            pop = reader.load_population(snapshot['snapshotId'])
            np.testing.assert_allclose(pop.objs, problem.evaluate(pop.decs).objs)
        best = reader.load_last_best()
        np.testing.assert_allclose(best.decs, result.bestDecs)
    finally:
        reader.close()
    payload = algorithm.saveResult()
    np.testing.assert_allclose(payload['bestDecs'], result.bestDecs)
    np.testing.assert_allclose(payload['bestObjs'], problem.evaluate(result.bestDecs).objs)
    np.testing.assert_allclose(payload['bestObjHistory'], result.history.bestObjHistory)


def test_constraint_evaluation_receives_real_values_without_mutating_unit_population():
    from UQPyL.problem import ModelProblem
    received = []
    def simulate(X):
        received.append(X.copy())
        return np.sum(X, axis=1, keepdims=True)
    problem = ModelProblem(nInput=3, nObj=1, nCon=1, lb=[100, 2, -7], ub=[200, 5, 8],
                           varType=[0, 1, 2], varSet={2: [10, 20, 70]}, simFunc=simulate,
                           objFunc=lambda X, context: context.sims,
                           conFunc=lambda X, context: X[:, 2:3]-20)
    algorithm = GA(nPop=3, maxFEs=3, **QUIET)
    algorithm.setup(problem, 1)
    original = np.array([[.2, .1, .1], [.4, .4, .5], [.9, .9, .9]])
    pop = Population(original)
    algorithm.evaluate(pop)
    np.testing.assert_array_equal(pop.decs, original)
    np.testing.assert_allclose(received[0], [[120, 2, 10], [140, 3, 20], [190, 5, 70]])
    np.testing.assert_allclose(pop.objs, [[132], [163], [265]])
    np.testing.assert_allclose(pop.cons, [[-10], [0], [50]])


def test_unit_conversion_roundoff_at_mixed_boundary():
    problem = Problem(nInput=2, nObj=1, lb=[-.1, 0], ub=[.2, 1], varType=[0, 2],
                      varSet={1: [10, 20]}, objFunc=lambda X: X[:, :1])
    np.testing.assert_allclose(problem.canonicalize_unit([[1, 1]]), [[1, .75]])


def test_surrogate_maximization_subproblem_uses_minimization_of_internal_scores():
    problem = Problem(nInput=1, nObj=1, lb=10., ub=20., optType='max', objFunc=lambda X: X)
    class ExactSurrogate:
        def fit(self, X, Y):
            np.testing.assert_allclose(Y, -(10+10*X))
            return self
        def predict(self, X):
            return -(10+10*X)
    class Inner:
        verboseFlag = logFlag = saveFlag = False
        def run(self, subProblem, **kwargs):
            np.testing.assert_array_equal(subProblem.lb, [[0.]])
            np.testing.assert_array_equal(subProblem.ub, [[1.]])
            np.testing.assert_array_equal(subProblem.opt, [1])
            np.testing.assert_allclose(subProblem.evaluate([[0.], [1.]]).objs, [[-10], [-20]])
            class Result:
                bestDecs = np.array([[.9]])
            return Result()
    algorithm = ASMO(nInit=2, maxFEs=3, surrogate=ExactSurrogate(), optimizer=Inner(), **QUIET)
    result = algorithm.run(problem, initialPop=[[11.], [12.]], seed=1, oneStep=True)
    np.testing.assert_allclose(result.bestDecs, [[19.]])
    np.testing.assert_allclose(result.bestObjs, [[19.]])
    np.testing.assert_allclose(algorithm.state.bestObjs, [[-19.]])
    for entry in result.history.populations:
        np.testing.assert_allclose(entry["objs"], problem.evaluate(entry["decs"]).objs)


def test_multiobjective_mixed_directions_export_original_objectives():
    problem = mixedProblem(2, optType=['min', 'max'])
    result = NSGAII(nPop=12, maxFEs=24, maxIters=1, **QUIET).run(problem, seed=42)
    np.testing.assert_allclose(result.bestObjs, problem.evaluate(result.bestDecs).objs)
    for snapshot in result.history.populations:
        np.testing.assert_allclose(snapshot['objs'], problem.evaluate(snapshot['decs']).objs)
