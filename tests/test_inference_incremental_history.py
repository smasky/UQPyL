"""Inference incremental history.

Migrated from test_review_c13_c14.py; original regression provenance is retained below.
"""

from copy import deepcopy
import numpy as np
import pytest
from UQPyL.inference import MH, InfReader
from UQPyL.inference.chain import Chain
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def assertCollectedState(state, chains, problem):
    """Independent full-history oracle for the incremental summaries."""
    draw = min(chain.count for chain in chains)
    decs = np.stack([chain.decs[:draw] for chain in chains])
    objs = np.stack([chain.objs[:draw] for chain in chains])
    cons = np.stack([chain.cons[:draw] for chain in chains])
    logs = np.stack([chain.logProb[:draw] for chain in chains])
    accepted = np.stack([chain.accepted[:draw] for chain in chains])
    feasible = np.all(cons <= 0, axis=2)
    np.testing.assert_array_equal(state.decs, decs)
    np.testing.assert_array_equal(state.objs, objs)
    np.testing.assert_array_equal(state.cons, cons)
    np.testing.assert_array_equal(state.logProb, logs)
    np.testing.assert_array_equal(state.accepted, accepted)
    np.testing.assert_array_equal(state.feasibleMask, feasible)
    np.testing.assert_allclose(state.meanLogProb, np.nanmean(logs), rtol=1e-14)
    np.testing.assert_allclose(state.feasibleRate, feasible.mean())
    expectedAcceptance = accepted[:, 1:].mean(axis=1) if draw > 1 else np.ones(len(chains))
    np.testing.assert_array_equal(state.acceptanceRate, expectedAcceptance)
    flatDecs, flatObjs, flatCons = decs.reshape(-1, 2), objs.reshape(-1, 1), cons.reshape(-1, 2)
    indices = np.flatnonzero(feasible.ravel()) if np.any(feasible) else np.arange(flatObjs.size)
    best = indices[np.argmin(flatObjs[indices, 0])]
    np.testing.assert_array_equal(state.bestDecs, flatDecs[best : best + 1])
    np.testing.assert_array_equal(state.bestObjs, flatObjs[best : best + 1] * problem.opt)
    np.testing.assert_array_equal(state.bestCons, flatCons[best : best + 1])
    assert state.bestFeasible == bool(np.any(feasible))
    sample = flatDecs[feasible.ravel()] if np.any(feasible) else flatDecs
    mean, std = state.decisionMoments()
    np.testing.assert_allclose(mean, np.mean(sample, axis=0), atol=1e-14)
    np.testing.assert_allclose(std, np.std(sample, axis=0), atol=1e-14)


# Regression source: test_review_c13_c14.py::testIncrementalInferenceMatchesFullHistory
@pytest.mark.parametrize("direction", ["min", "max"])
@pytest.mark.parametrize("feasibility", ["none", "all", "later"])
def testIncrementalInferenceMatchesFullHistory(direction, feasibility):
    problem = Problem(
        nInput=2, nObj=1, nCon=2, lb=0, ub=10, objFunc=lambda x: x[:, :1], conFunc=lambda x: x - 5, optType=direction
    )
    method = MH(nChains=3, maxIters=8, warmUp=0, **QUIET)
    method.setup(problem, seed=1)
    chains = [Chain(2, 1, 2, 8) for _ in range(3)]
    # Deliberate objective ties across chains and draws exercise first-occurrence ordering.
    scores = np.array([[3, 2, 1, 1, 0, 0, -1, -1], [1, 1, 0, 0, 0, -1, -1, -1], [2, 0, 0, 0, 0, 0, 0, 0]])
    for stop in (1, 3, 4, 8):
        for chainId, chain in enumerate(chains):
            while chain.count < stop:
                draw = chain.count
                isFeasible = feasibility == "all" or (feasibility == "later" and draw >= 3 and chainId != 2)
                chain.add(
                    [chainId + 1, draw + 1],
                    [scores[chainId, draw]],
                    [-1, -1 if isFeasible else 1],
                    logProb=np.nan if chainId == 0 and draw == 0 else -scores[chainId, draw],
                    accepted=(draw + chainId) % 3 != 0,
                )
            # Also check asynchronous chain counts: only complete draws are visible.
            if min(item.count for item in chains):
                method.state.update(chains, problem, stop * 3, stop - 1)
                assertCollectedState(method.state, chains, problem)
        saved = deepcopy(method.state.buildSnapshot(stop * 3, stop - 1))
        method.state.update(chains, problem, stop * 3, stop - 1)
        assert method.state.buildSnapshot(stop * 3, stop - 1) == saved
    result = method.buildResult()
    result.decs[:] = -999
    assertCollectedState(method.state, chains, problem)
    replacement = [Chain(2, 1, 2, 2)]
    replacement[0].add([9, 8], [42], [-1, -1], logProb=-42)
    method.state.update(replacement, problem, 1, 0)
    assertCollectedState(method.state, replacement, problem)


# Regression source: test_review_c13_c14.py::testInferenceHistoryWorkIsLinearAndSnapshotsAreLight
@pytest.mark.parametrize("draws", [20, 40, 80])
@pytest.mark.parametrize("save,log", [(False, False), (True, False), (False, True)])
def testInferenceHistoryWorkIsLinearAndSnapshotsAreLight(draws, save, log, tmp_path, monkeypatch):
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, optType="max", objFunc=lambda x: np.sum(x * x, axis=1)[:, None])
    problem.workDir = str(tmp_path)
    method = MH(
        nChains=2, maxIters=draws, warmUp=0, verboseFlag=False, saveFlag=save, saveFreq=1, logFlag=log, verboseFreq=1
    )
    originalCollect, originalDecode = method.state._collect, method._decodeDecs
    originalBuild = method.state.buildResult
    insideCollect = False
    decodedRows, builds = [], []

    def collect(*args):
        nonlocal insideCollect
        insideCollect = True
        try:
            return originalCollect(*args)
        finally:
            insideCollect = False

    def decode(x):
        if insideCollect:
            decodedRows.append(len(x))
        return originalDecode(x)

    def build():
        builds.append(True)
        return originalBuild()

    monkeypatch.setattr(method.state, "_collect", collect)
    monkeypatch.setattr(method, "_decodeDecs", decode)
    monkeypatch.setattr(method.state, "buildResult", build)
    result = method.run(problem, seed=4)
    assert sum(decodedRows) == 2 * draws and max(decodedRows) == 2
    assert len(builds) == 1
    assert result.decs.shape == (2, draws, 2)
    for draw, snapshot in enumerate(result.history.snapshots):
        assert snapshot["meanLogProb"] == pytest.approx(result.logProb[:, : draw + 1].mean())
    if save:
        with InfReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
            loaded = reader.load_result()
            np.testing.assert_array_equal(loaded.objs, result.objs)
            for snapshot in reader.list_snapshots():
                draw = snapshot["iter"]
                assert snapshot["meanLogProb"] == pytest.approx(result.logProb[:, : draw + 1].mean())
                for member in reader.load_snapshot_members(snapshot["snapshotId"]):
                    chain = member["chain"]
                    np.testing.assert_array_equal(member["decs"], result.decs[chain, draw])
                    np.testing.assert_array_equal(member["objs"], result.objs[chain, draw])
