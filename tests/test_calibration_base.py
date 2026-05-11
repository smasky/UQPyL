from pathlib import Path
import shutil
import sqlite3
import uuid

import numpy as np

from UQPyL.calibration import CalReader, CalibrationABC, SqliteStorage
from UQPyL.problem import ModelProblem


class DummyCalibration(CalibrationABC):
    name = "DummyCalibration"

    def _runCore(self, problem, X):
        sim_valid = self.evaluate(X, validOnly=True)
        sim_full = self.evaluate(X, validOnly=False)
        self.state.diagnostics["validShape"] = sim_valid.shape
        self.state.diagnostics["fullShape"] = sim_full.shape
        self.recordBest(np.asarray(X)[0:1], sim_full[0:1])


def _sqlite_simf(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


def test_calibration_base_runs_on_model_problem():
    obs = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[False, True], [False, False]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 2))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 0, 1] = X[:, 0] + 10.0
        sim[:, 1, 0] = X[:, 1]
        sim[:, 1, 1] = X[:, 1] + 10.0
        return sim

    problem = ModelProblem(
        nInput=2,
        ub=1.0,
        lb=0.0,
        simFunc=simf,
        obs=obs,
        mask=mask,
        simLabels=["A", "B"],
        name="HBV",
    )

    method = DummyCalibration()
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    res = method.run(problem, X)

    assert res.method == "DummyCalibration"
    assert res.problemName == "HBV"
    assert res.nTime == 2
    assert res.nSeries == 2
    assert res.nObs == 4
    assert np.array_equal(res.obs, obs)
    assert np.array_equal(res.mask, mask)
    assert res.simLabels == ["A", "B"]
    assert res.bestDecs.shape == (1, 2)
    assert res.bestSim.shape == (1, 4)
    assert res.posteriorDecs is None
    assert res.posteriorSims is None
    assert res.behavioralDecs is None
    assert res.behavioralSims is None
    assert res.eliteDecs is None
    assert res.eliteSims is None
    assert res.diagnostics["fullShape"] == (2, 4)
    assert res.diagnostics["validShape"] == (2, 3)


def test_calibration_base_constructor_keeps_common_runtime_flags():
    method = DummyCalibration(verboseFlag=True, verboseFreq=5, saveFlag=True, logFlag=True)

    assert method.verboseFlag is True
    assert method.verboseFreq == 5
    assert method.saveFlag is True
    assert method.logFlag is True
    assert method.get("verboseFlag", "verboseFreq", "saveFlag", "logFlag") == (True, 5, True, True)


def test_calibration_result_summary_and_log_file():
    obs = np.array([[1.0], [2.0]])

    def simf(X):
        X = np.atleast_2d(X)
        sim = np.zeros((X.shape[0], 2, 1))
        sim[:, 0, 0] = X[:, 0]
        sim[:, 1, 0] = X[:, 1]
        return sim

    work_dir = Path(".cache") / "calibration_tests" / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        problem = ModelProblem(
            nInput=2,
            ub=1.0,
            lb=0.0,
            simFunc=simf,
            obs=obs,
            simLabels=["Q"],
            name="HBV",
        )
        problem.workDir = str(work_dir)

        method = DummyCalibration(logFlag=True)
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        res = method.run(problem, X)

        summary = res.summary()
        assert summary["method"] == "DummyCalibration"
        assert summary["problem_name"] == "HBV"
        assert summary["metric"] == "rmse"
        assert summary["best_x"].shape == (2,)

        logFiles = list((work_dir / "Result").glob("*.log"))
        assert logFiles
        text = logFiles[0].read_text(encoding="utf-8")
        assert "DummyCalibration finished" in text
        assert "bestX" in text
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_calibration_save_flag_writes_sqlite_result():
    work_dir = Path(".cache") / "calibration_sqlite_tests" / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        problem = ModelProblem(
            nInput=2,
            ub=1.0,
            lb=0.0,
            simFunc=_sqlite_simf,
            obs=np.array([[1.0], [2.0]]),
            simLabels=["Q"],
            name="HBV",
        )
        problem.workDir = str(work_dir)

        method = DummyCalibration(saveFlag=True)
        res = method.run(problem, np.array([[0.1, 0.2], [0.3, 0.4]]))

        dbFiles = list((work_dir / "Result").glob("*.sqlite3"))
        assert len(dbFiles) == 1
        assert method.runId is not None
        assert res.runId == method.runId
        assert method.session is None
        assert SqliteStorage is not None

        conn = sqlite3.connect(dbFiles[0])
        conn.row_factory = sqlite3.Row
        try:
            run = conn.execute("SELECT * FROM run LIMIT 1").fetchone()
            assert run["runId"] == method.runId
            assert run["method"] == "DummyCalibration"
            assert run["problem"] == "HBV"
            assert run["status"] == "finished"
            assert run["nInput"] == 2
            assert run["nTime"] == 2
            assert run["nSeries"] == 1
            assert run["nObs"] == 2
            assert run["runtime"] >= 0.0

            artifactNames = {
                row["name"]
                for row in conn.execute("SELECT name FROM artifact ORDER BY name").fetchall()
            }
            assert {"result", "obs", "mask", "bestDecs", "bestSim", "settings"} <= artifactNames
            param = conn.execute(
                "SELECT value FROM runParam WHERE name = ?",
                ("saveFlag",),
            ).fetchone()
            assert param["value"] == "True"
        finally:
            conn.close()

        runs = CalReader.list_runs(work_dir)
        assert len(runs) == 1
        assert runs[0]["run_id"] == method.runId
        assert runs[0]["method"] == "DummyCalibration"
        assert runs[0]["problem"] == "HBV"
        assert runs[0]["status"] == "finished"

        with CalReader(dbFiles[0]) as reader:
            summary = reader.get_run_summary()
            assert summary["run_id"] == method.runId
            assert summary["n_time"] == 2
            assert summary["n_series"] == 1
            assert summary["n_obs"] == 2
            assert "result" in summary["artifact_names"]
            loaded = reader.load_result()
            assert loaded.runId == method.runId
            assert loaded.summary()["run_id"] == method.runId
            assert np.array_equal(loaded.bestDecs, res.bestDecs)

        assert res.bestDecs.shape == (1, 2)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
