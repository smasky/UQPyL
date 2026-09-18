import numpy as np
import pytest

from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Guass
from UQPyL.optimization.soea import GA


class ProbeOptimizer:
    type = 'MP'

    def __init__(self):
        self.starts = []
        self.values = []
        self.seeds = []

    def run(self, problem, xInit=None, seed=None):
        self.starts.append(xInit.copy())
        self.seeds.append(seed)
        self.values.append(float(problem.objFunc(xInit)))
        # Deliberately stale score: selection must use the returned point.
        return xInit, -len(self.starts) * 1e100


def makeModel(family, optimizer, count=None, log=True):
    attr = dict(lb=.1, ub=3., type='float', log=log)
    if family is GPR:
        return GPR(kernel=RBF(length_scale=[.3, .7], heterogeneous=True, length_attr=attr),
                   C=1e-6, C_attr=None, optimizer=optimizer, nRestartTimes=count)
    return KRG(kernel=Guass(theta=[.3, .7], heterogeneous=True, theta_attr=attr),
               optimizer=optimizer, nRestartTimes=count)


@pytest.mark.parametrize('family', [GPR, KRG])
@pytest.mark.parametrize('log', [True, False])
@pytest.mark.parametrize('count', [None, 0, np.int64(2)])
def testStartsSelectionAndReproducibility(family, log, count):
    x = np.random.default_rng(8).uniform(size=(12, 2))
    y = np.sin(5*x[:, :1]) + x[:, 1:]
    globalState = np.random.get_state()
    runs = []
    for _ in range(2):
        probe = ProbeOptimizer()
        model = makeModel(family, probe, count, log)
        model.rng = np.random.default_rng(42)
        model.fit(x, y)
        assert len(probe.starts) == (5 if count is None else count+1)
        np.testing.assert_allclose(probe.starts[0], np.log([.3, .7]) if log else [.3, .7])
        assert len(set(probe.seeds)) == len(probe.seeds)
        for start, seed in zip(probe.starts[1:], probe.seeds[1:]):
            lower, upper = (np.log(.1), np.log(3.)) if log else (.1, 3.)
            np.testing.assert_array_equal(start, np.random.default_rng(seed).uniform(lower, upper, 2))
        selected = probe.starts[np.argmin(probe.values)]
        actual = model.setting.get('l' if family is GPR else 'theta')
        np.testing.assert_allclose(actual, np.exp(selected) if log else selected)
        assert model.fitState['objective'] == pytest.approx(min(probe.values))
        runs.append((probe.starts, model.predict(x)))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
    after = np.random.get_state()
    np.testing.assert_array_equal(globalState[1], after[1])
    assert globalState[2:] == after[2:]


@pytest.mark.parametrize('family', [GPR, KRG])
@pytest.mark.parametrize('count', [-1, 1.5, 2., True, np.bool_(False), '4'])
def testInvalidRestartCount(family, count):
    with pytest.raises(ValueError, match='non-negative integer'):
        family(nRestartTimes=count)


@pytest.mark.parametrize('family', [GPR, KRG])
def testEaDefaultUnchanged(family):
    assert family(optimizer=GA()).nRes == 1


@pytest.mark.parametrize('family', [GPR, KRG])
def testFixedParametersSkipOptimizer(family):
    probe = ProbeOptimizer()
    model = (GPR(kernel=RBF(length_scale=.3, length_attr=None), C_attr=None, optimizer=probe)
             if family is GPR else KRG(kernel=Guass(theta=1., theta_attr=None), optimizer=probe))
    x = np.linspace(0, 1, 8)[:, None]
    model.fit(x, np.sin(x))
    assert not probe.starts
    assert np.isfinite(model.predict(x)).all()


@pytest.mark.parametrize('family', [GPR, KRG])
def testAllNonfiniteCandidatesFailClearly(family):
    model = makeModel(family, ProbeOptimizer(), 2)
    method = '_objfunc' if family is GPR else '_objFunc'
    setattr(model, method, lambda *args, **kwargs: np.nan)
    x = np.random.default_rng(4).uniform(size=(8, 2))
    with pytest.raises(RuntimeError, match='No restart'):
        model.fit(x, x[:, :1])


@pytest.mark.parametrize('family', [GPR, KRG])
def testInvalidFirstCandidateDoesNotPoisonSelection(family):
    class InvalidFirst(ProbeOptimizer):
        def run(self, problem, xInit=None, seed=None):
            point, value = super().run(problem, xInit, seed)
            if len(self.starts) == 1:
                point[:] = np.nan
            return point, value

    probe = InvalidFirst()
    model = makeModel(family, probe, 2)
    x = np.random.default_rng(8).uniform(size=(12, 2))
    model.fit(x, np.sin(x[:, :1]))
    assert len(probe.starts) == 3
    assert model.fitState['objective'] == pytest.approx(min(probe.values[1:]))
