"""Calibration evaluation and storage.

Migrated from test_remaining_review.py; original regression provenance is retained below.
"""

import numpy as np
import pytest


# Regression source: test_remaining_review.py::testCalibrationReusesSimulationBatches
@pytest.mark.parametrize("name,count", [("ES", 2), ("IES", 4), ("SUFI2", 1)])
def testCalibrationReusesSimulationBatches(name, count):
    import UQPyL.calibration as calibration
    from UQPyL.problem import ModelProblem

    calls = []

    def simulate(X):
        values = np.column_stack([X[:, 0], X[:, 1] ** 2, X[:, 0] + X[:, 1]])[:, :, None]
        calls.append(values.copy())
        return (values).reshape(len(X), -1)

    problem = ModelProblem(
        nInput=2,
        lb=0.0,
        ub=3.0,
        simFunc=simulate,
        obs=np.array([1.0, 2.0, 0.0]),
        mask=np.array([False, False, True]),
    )
    method = getattr(calibration, name)(**({"maxIters": 3} if name == "IES" else {}))
    result = method.run(
        problem, np.array([[0.5, 0.8], [1.0, 1.5], [2.0, 2.0]]), **({"eliteSize": 2} if name == "SUFI2" else {})
    )
    assert len(calls) == count
    if name != "SUFI2":
        np.testing.assert_array_equal(result.posteriorSims, calls[-1][:, :, 0])
        np.testing.assert_allclose(result.diagnostics["scores"], method.score(result.posteriorSims))


# Regression source: test_remaining_review.py::testCalibrationStorageAndSummaryAvoidDuplicatePayloads
def testCalibrationStorageAndSummaryAvoidDuplicatePayloads(tmp_path, monkeypatch):
    from UQPyL.calibration import GLUE, CalReader
    from UQPyL.problem import ModelProblem

    problem = ModelProblem(
        nInput=1, lb=0.0, ub=1.0, obs=np.ones(30), simFunc=lambda X: (np.repeat(X[:, :, None], 30, axis=1)).reshape(len(X), -1)
    )
    problem.workDir = str(tmp_path)
    result = GLUE(saveFlag=True).run(problem, np.linspace(0, 1, 100)[:, None], threshold=10.0)
    with CalReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        sizes = dict(reader.conn.execute("SELECT name, length(payload) FROM artifact"))
        assert set(sizes) == {"result", "summary"}
        assert sum(sizes.values()) < 1.1 * sizes["result"]

        def unexpected(*args, **kwargs):
            raise AssertionError("Summary must not load full results")

        monkeypatch.setattr(reader, "get_artifacts", unexpected)
        monkeypatch.setattr(reader, "load_result", unexpected)
        assert reader.get_run_summary()["best_score"] == result.summary()["best_score"]
