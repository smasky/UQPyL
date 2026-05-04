import numpy as np

from UQPyL.problem import ProblemABC
from UQPyL.analysis import DeltaTest
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _zero_obj(x):
    return 0.0


def test_delta_private_cal_delta_smoke():
    dt = DeltaTest(verboseFlag=False, logFlag=False, saveFlag=False)
    # simple, deterministic points
    X = np.array([[0.0], [0.5], [1.0]])
    Y = np.array([[0.0], [0.5], [1.0]])
    val = dt._cal_delta(X, Y, nNeighbors=1)
    assert np.isfinite(val)


def test_delta_find_comb_vio_small_problem():
    # nInput=3 -> brute force 2^3 = 8 combos, fast and deterministic
    problem = Problem(nInput=3, nObj=1, ub=1.0, lb=0.0, xLabels=["x1", "x2", "x3"], objFunc=_zero_obj)
    dt = DeltaTest(nNeighbors=1, verboseFlag=False, logFlag=False, saveFlag=False)

    # make X,Y so x1 dominates
    rng = np.random.default_rng(123)
    X = rng.random((30, 3))
    Y = X[:, [0]]  # only depends on x1

    labels = dt.findCombVio(problem, X, Y)
    assert isinstance(labels, list)
    assert "x1" in labels


def test_delta_find_comb_ea_monkeypatched(monkeypatch):
    # avoid running real GA; just ensure control flow reaches ga.run and returns its result
    problem = Problem(nInput=3, nObj=1, ub=1.0, lb=0.0, xLabels=["x1", "x2", "x3"], objFunc=_zero_obj)
    dt = DeltaTest(nNeighbors=1, verboseFlag=False, logFlag=False, saveFlag=False)

    rng = np.random.default_rng(123)
    X = rng.random((20, 3))
    Y = X[:, [0]]

    class DummyGA:
        def __init__(self, maxFEs=None, verboseFlag=None, saveFlag=None):
            self.maxFEs = maxFEs

        def run(self, problem):
            # sanity: EA problem is binary/integer selection
            assert problem.nInput == 3
            assert problem.nOutput == 1
            return {"ok": True, "maxFEs": self.maxFEs}

    import UQPyL.analysis.delta as delta_mod

    monkeypatch.setattr(delta_mod, "GA", DummyGA)

    res = dt.findCombEA(problem, X, Y, FEs=10, verboseFlag=False, saveFlag=False)
    assert res["ok"] is True
    assert res["maxFEs"] == 10


def test_delta_analyze_uses_relative_delta_change():
    @ProblemABC.singleFunc
    def obj_func(x):
        return float(x[0])

    problem = Problem(nInput=3, nObj=1, ub=1.0, lb=0.0, xLabels=["x1", "x2", "x3"], objFunc=obj_func)
    dt = DeltaTest(nNeighbors=1, verboseFlag=False, logFlag=False, saveFlag=False)

    rng = np.random.default_rng(123)
    X = rng.random((40, 3))
    Y = X[:, [0]]

    res = dt.analyze(problem, X, Y=Y)
    s1Metric = next(metric for metric in res.metrics if metric.name == "S1")
    s1NormMetric = next(metric for metric in res.metrics if metric.name == "S1_norm")

    assert np.all(np.isfinite(s1Metric.values))
    assert np.all(np.isfinite(s1NormMetric.values))
    assert np.argmax(s1Metric.values[0]) == 0


