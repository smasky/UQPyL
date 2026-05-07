import numpy as np

from UQPyL.doe import FASTDesign, LHS, SaltelliDesign
from UQPyL.analysis import FAST, RBDFAST, Sobol
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _ishigami_obj(x, a=7.0, b=0.1):
    x = np.asarray(x)
    x1, x2, x3 = x[0], x[1], x[2]
    return float(np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1))


def _make_ishigami_problem():
    return Problem(
        nInput=3,
        nObj=1,
        lb=[-np.pi, -np.pi, -np.pi],
        ub=[np.pi, np.pi, np.pi],
        objFunc=_ishigami_obj,
        optType="min",
        xLabels=["x1", "x2", "x3"],
        name="Ishigami",
    )


def test_sobol_ishigami_matches_theoretical_indices():
    problem = _make_ishigami_problem()
    sob = Sobol(verboseFlag=False, logFlag=False, saveFlag=False)
    X, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, 1024, seed=123)
    Y = problem.evaluate(X, target="objs").objs

    res = sob.analyze(problem, X, Y=Y, meta=meta, target="objs", index="all")
    s1Metric = res.getMetric("S1")
    stMetric = res.getMetric("ST")

    expectedS1 = np.array([0.3139, 0.4424, 0.0])
    expectedST = np.array([0.5576, 0.4424, 0.2437])

    assert np.allclose(s1Metric.values[0], expectedS1, atol=0.08)
    assert np.allclose(stMetric.values[0], expectedST, atol=0.08)


def test_fast_ishigami_is_reasonably_close_to_theoretical_indices():
    problem = _make_ishigami_problem()
    fast = FAST(verboseFlag=False, logFlag=False, saveFlag=False)
    X, meta = FASTDesign(M=4).sampleWithMeta(problem, 1025, seed=123)
    Y = problem.evaluate(X, target="objs").objs

    res = fast.analyze(problem, X, Y=Y, meta=meta, target="objs", index="all")
    s1Metric = res.getMetric("S1")
    stMetric = res.getMetric("ST")

    expectedS1 = np.array([0.3139, 0.4424, 0.0])
    expectedST = np.array([0.5576, 0.4424, 0.2437])

    assert np.allclose(s1Metric.values[0], expectedS1, atol=0.12)
    assert np.allclose(stMetric.values[0], expectedST, atol=0.15)


def test_rbd_fast_ishigami_recovers_first_order_ranking():
    problem = _make_ishigami_problem()
    rbd = RBDFAST(M=4, verboseFlag=False, logFlag=False, saveFlag=False)
    X = LHS("classic").sample(problem, 4096, seed=123)
    Y = problem.evaluate(X, target="objs").objs

    res = rbd.analyze(problem, X, Y=Y, target="objs", index="all")
    s1Metric = res.getMetric("S1")

    expectedS1 = np.array([0.3139, 0.4424, 0.0])

    assert np.allclose(s1Metric.values[0], expectedS1, atol=0.12)
    assert np.argmax(s1Metric.values[0]) == 1
    assert np.argmin(s1Metric.values[0]) == 2
