"""Reference fixtures and behavioral tests for explicit MCMC diagnostics."""

from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from UQPyL.inference import MH, InfReader
from UQPyL.inference.diagnostics import computeChainDiagnostics
from UQPyL.problem import Problem


DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "inference_diagnostics_arviz_022.json").read_text())
METRICS = ("split_rhat", "rhat", "ess_bulk", "ess_tail")


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def testMatchesPinnedArvizReference(case):
    with np.load(DATA / "inference_diagnostics_arviz_022.npz", allow_pickle=False) as fixture:
        samples = fixture[case["name"]]
    report = computeChainDiagnostics(samples[:, :, None])
    for key, expected in case["expected"].items():
        assert report[key]["status"] == ["available"]
        np.testing.assert_allclose(report[key]["values"], [expected], atol=2e-10, rtol=2e-10)
    if samples.shape[0] == 1:
        for key in ("rhat", "split_rhat"):
            assert report[key]["status"] == ["insufficient_chains"]
            assert report[key]["values"] == [None]


@pytest.mark.parametrize("scale", [2.0**-800, 2.0**800, -(2.0**800), 2.0**-300, -(2.0**-300)])
def testExtremePowerOfTwoScalesPreserveDiagnosticsWithoutMutatingInput(scale):
    # Exact binary scaling preserves ties. Decimal rescaling can change the
    # last-bit tie of the two central folded values, also in ArviZ itself.
    samples = np.random.default_rng(16).normal(size=(4, 256, 2))
    expected = computeChainDiagnostics(samples)
    scaled = samples * scale
    saved = scaled.copy()
    actual = computeChainDiagnostics(scaled)
    for key in METRICS:
        assert actual[key]["status"] == ["available", "available"]
        np.testing.assert_allclose(actual[key]["values"], expected[key]["values"], rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(scaled, saved)


@pytest.mark.parametrize(
    "shape,status",
    [
        ((0, 16, 2), "insufficient_chains"),
        ((2, 0, 2), "insufficient_draws"),
        ((2, 3, 2), "insufficient_draws"),
        ((2, 16, 2), "constant_chain"),
    ],
)
def testUnavailableShapesAndConstantChains(shape, status):
    report = computeChainDiagnostics(np.zeros(shape))
    for key in METRICS:
        assert report[key]["values"] == [None, None]
        assert report[key]["status"] == [status, status]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "value,status", [(np.nan, "nonfinite"), (np.inf, "nonfinite"), ("red", "non_numeric"), (1j, "non_numeric")]
)
def testInvalidColumnDoesNotHideValidColumns(value, status):
    normal = np.random.default_rng(5).normal(size=(4, 128))
    samples = np.empty((4, 128, 2), dtype=object)
    samples[:, :, 0] = normal
    samples[:, :, 1] = value
    report = computeChainDiagnostics(samples)
    for key in METRICS:
        assert report[key]["status"] == ["available", status]
        assert report[key]["values"][1] is None
    json.dumps(report, allow_nan=False)


def testOneFrozenChainIsNotReportedAsFullSampleEss():
    samples = np.random.default_rng(5).normal(size=(4, 128, 1))
    samples[0] = 0
    for metric in (computeChainDiagnostics(samples)[key] for key in METRICS):
        assert metric["status"] == ["constant_chain"] and metric["values"] == [None]


def testDiscreteDegenerateFoldAndTailHaveExplicitStatus():
    samples = np.tile([0.0, 1.0], (4, 64))[:, :, None]
    report = computeChainDiagnostics(samples)
    assert report["rhat"]["values"] == [None]
    assert report["rhat"]["status"] == ["constant_folded"]
    assert report["ess_tail"]["values"] == [None]
    assert report["ess_tail"]["status"] == ["constant_tail"]
    assert report["ess_bulk"]["status"] == ["available"]
    assert report["ess_bulk"]["values"][0] > samples.size


def testFoldedRhatDetectsScaleMismatchMissedByClassicalStatistic():
    samples = np.random.default_rng(77).normal(size=(4, 4000, 1))
    samples *= np.array([0.1, 0.1, 10.0, 10.0])[:, None, None]
    report = computeChainDiagnostics(samples)
    assert report["split_rhat"]["values"][0] < 1.01
    assert report["rhat"]["values"][0] > 1.3


@pytest.mark.parametrize("rho", [0.8, -0.5])
def testArEssAgreesWithAsymptoticCorrelationScale(rho):
    samples = np.random.default_rng(14).normal(size=(4, 8192))
    for index in range(1, samples.shape[1]):
        samples[:, index] = rho * samples[:, index - 1] + np.sqrt(1 - rho**2) * samples[:, index]
    report = computeChainDiagnostics(samples[:, :, None])
    expected = samples.size * (1 - rho) / (1 + rho)
    assert report["ess_bulk"]["values"][0] == pytest.approx(expected, rel=0.3)


def testDiagnosticsUseFftAndDoNotReorderOrConsumeGlobalRandomness(monkeypatch):
    from UQPyL.inference import diagnostics

    samples = np.random.default_rng(22).normal(size=(4, 1025, 2))
    saved = samples.copy()
    calls = []
    original = diagnostics.rfft

    def counted(values, **kwargs):
        calls.append(values.shape)
        return original(values, **kwargs)

    monkeypatch.setattr(diagnostics, "rfft", counted)
    state = pickle.dumps(np.random.get_state())
    computeChainDiagnostics(samples)
    assert calls == [(8, 512)] * 6  # Bulk plus two quantiles, per variable.
    np.testing.assert_array_equal(samples, saved)
    assert pickle.dumps(np.random.get_state()) == state


def testResultAndReaderDiagnosticsAreExplicitAndIndependent(tmp_path):
    calls = []

    def objective(x):
        calls.append(len(x))
        return x**2

    problem = Problem(nInput=1, nObj=1, lb=-3, ub=3, objFunc=objective)
    problem.workDir = str(tmp_path)
    method = MH(nChains=4, warmUp=0, maxIters=100, saveFlag=True, verboseFlag=False)
    result = method.run(problem, seed=2)
    assert result.diagnostics["chains"] == {"status": "not_computed"}
    samples, count, rngState = result.decs.copy(), sum(calls), deepcopy(method.rng.bit_generator.state)
    report = result.computeDiagnostics()
    assert sum(calls) == count and method.rng.bit_generator.state == rngState
    assert result.stopReason == "max_iters"
    assert method.state.diagnostics["chains"] == {"status": "not_computed"}
    np.testing.assert_array_equal(result.decs, samples)
    for key in ("rhat", "ess_bulk", "ess_tail"):
        assert report[key]["status"] == ["available"]
    report["ess_bulk"]["values"][0] = -1
    assert result.diagnostics["chains"]["ess_bulk"]["values"][0] > 0
    with InfReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
        assert loaded.diagnostics["chains"] == {"status": "not_computed"}
        assert loaded.computeDiagnostics() == result.diagnostics["chains"]
        assert reader.load_result().diagnostics["chains"] == {"status": "not_computed"}


def testMalformedShapeIsRejected():
    with pytest.raises(ValueError, match="shape"):
        computeChainDiagnostics(np.zeros((4, 100)))
