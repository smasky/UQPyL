from contextlib import nullcontext

import numpy as np
import pytest

from UQPyL.inference import DREAM_ZS
from UQPyL.problem import Problem


class ArchiveProbe(DREAM_ZS):
    """Observe the real run loop against an independently copied event history."""

    def __init__(self, *, acceptance=None, controlled=False, **kwargs):
        super().__init__(nChains=3, verboseFlag=False, logFlag=False, saveFlag=False, **kwargs)
        self.acceptance = acceptance
        self.controlled = controlled

    def initialSampling(self, problem, nChains):
        X, objs, cons = super().initialSampling(problem, nChains)
        self.expectedArchive = [row.copy() for row in X]
        self.retained = {}
        self.acceptCalls = 0
        self.proposalCalls = 0
        self.acceptedCount = 0
        self.frozenArchive = None
        return X, objs, cons

    def checkArchive(self, current, archive):
        capacity = self.get("nChains") * self.get("archSize")
        expectedCount = min(capacity, 3 * (1 + min(self.proposalCalls, self.get("warmUp"))))
        assert len(archive) == expectedCount
        for point in archive:
            assert any(np.array_equal(point, occupied) for occupied in self.expectedArchive)
        if self.frozenArchive is not None:
            np.testing.assert_array_equal(archive, self.frozenArchive)
        # Check even entries already evicted from the rolling archive.
        for point, snapshot in self.retained.values():
            np.testing.assert_array_equal(point, snapshot)
            assert not np.shares_memory(point, current)
        for index, point in enumerate(archive):
            assert not np.shares_memory(point, current)
            for other in archive[:index]:
                assert not np.shares_memory(point, other)
            self.retained.setdefault(id(point), (point, point.copy()))

    def f_prop_ratio(self, X_cur, archive, *args, **kwargs):
        if 0 < self.proposalCalls <= self.get("warmUp"):
            self.expectedArchive.extend(point.copy() for point in X_cur)
        if self.proposalCalls == self.get("warmUp"):
            self.frozenArchive = np.array(archive).copy()
        self.checkArchive(X_cur, archive)
        self.proposalCalls += 1
        self.lastCurrent, self.lastArchive = X_cur, archive
        if self.controlled:
            return 0.9 * X_cur + 0.02, np.zeros(3), np.zeros(3, dtype=int)
        return super().f_prop_ratio(X_cur, archive, *args, **kwargs)

    def accept(self, *args, **kwargs):
        if self.acceptance is None:
            accepted = super().accept(*args, **kwargs)
        else:
            accepted = (self.acceptance == "all"
                        or (self.acceptance == "alternating" and self.acceptCalls % 2 == 0))
        self.acceptCalls += 1
        if accepted:
            self.acceptedCount += 1
        return accepted

    def finalize(self):
        # Formal updates cannot alter the warm-up reservoir.
        self.checkArchive(self.lastCurrent, self.lastArchive)
        return super().finalize()


def flatProblem():
    return Problem(nInput=2, nObj=1, lb=-2., ub=2., objFunc=lambda X: np.zeros((len(X), 1)))


@pytest.mark.parametrize("warmUp", [0, 3])
@pytest.mark.parametrize("archSize", [1, 5])
@pytest.mark.parametrize("acceptance", ["all", "alternating", "none"])
def test_archive_records_occupation_states_then_freezes(warmUp, archSize, acceptance):
    method = ArchiveProbe(warmUp=warmUp, archSize=archSize, maxIters=5,
                          controlled=True, acceptance=acceptance)
    result = method.run(flatProblem(), gamma=0.1, seed=3)
    assert method.proposalCalls == warmUp + 4
    assert method.acceptCalls == 3 * method.proposalCalls
    assert len(method.expectedArchive) == 3 * (1 + warmUp)
    assert result.decs.shape == (3, 5, 2)
    assert np.isfinite(result.decs).all()
    if acceptance == "all":
        assert method.acceptedCount == method.acceptCalls
    elif acceptance == "none":
        assert method.acceptedCount == 0
    else:
        assert 0 < method.acceptedCount < method.acceptCalls
    # A future in-place update cannot rewrite any retained history point.
    method.lastCurrent[:] = 123.
    for point, snapshot in method.retained.values():
        np.testing.assert_array_equal(point, snapshot)


@pytest.mark.parametrize("ps", [0., 1.])
@pytest.mark.parametrize("warmUp", [0, 2])
@pytest.mark.parametrize("seed", [3, 7])
def test_real_de_and_snooker_runs_preserve_history_and_reproduce(ps, warmUp, seed):
    method = ArchiveProbe(warmUp=warmUp, archSize=2, maxIters=8, ps=ps, adpInterval=2)
    problem = flatProblem()
    before = np.random.get_state()
    with pytest.warns(RuntimeWarning, match="full-dimensional Gaussian refresh") if ps == 1 else nullcontext():
        first = method.run(problem, gamma=0.1, seed=seed)
    assert method.acceptedCount > 0
    with pytest.warns(RuntimeWarning, match="full-dimensional Gaussian refresh") if ps == 1 else nullcontext():
        second = method.run(problem, gamma=0.1, seed=seed)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    for field in ["decs", "objs", "logProb", "accepted"]:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert np.isfinite(first.decs).all()
    assert np.all(first.decs >= -2.) and np.all(first.decs <= 2.)
    np.testing.assert_array_equal(first.objs, np.zeros_like(first.objs))
