import numpy as np
import pytest
from pathlib import Path
import shutil
import json
import pickle
import sqlite3
import uuid

from UQPyL.doe import FASTDesign, LHS, MorrisDesign, SaltelliDesign
from UQPyL.analysis import FAST, RBDFAST, RSA, Sobol, MARS
from UQPyL.analysis.runtime import AnaReader
from UQPyL.core.runtime import build_db_path
from UQPyL.problem import ProblemABC
from UQPyL.problem.problem import Problem


@ProblemABC.singleFunc
def _nonlinear_obj(x):
    # non-constant objective to avoid zero variance in spectral methods
    x = np.asarray(x)
    return float(np.sum(np.sin(x) + 0.1 * x**2))


def _make_problem(n_input=3):
    return Problem(nInput=n_input, nObj=1, ub=1.0, lb=0.0, objFunc=_nonlinear_obj, optType="min")


def test_sobol_sample_validation_and_analyze_smoke():
    problem = _make_problem(3)
    sob = Sobol(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.warns(UserWarning):
        SaltelliDesign().sampleWithMeta(problem, 10)  # now warns instead of raising

    X, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 8, seed=123)
    assert X.shape == ((2 * problem.nInput + 2) * 8, problem.nInput)
    res = sob.analyze(problem, X, Y=None, meta=meta, target="objs", index="all")
    metricNames = {metric.name for metric in res.metrics}
    assert "S1" in metricNames and "ST" in metricNames and "S2" in metricNames

    X2, meta2 = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, 8, seed=123)
    res2 = sob.analyze(problem, X2, Y=None, meta=meta2, target="objs", index="all")
    metricNames2 = {metric.name for metric in res2.metrics}
    assert "S1" in metricNames2 and "ST" in metricNames2


def test_fast_sample_and_analyze_smoke():
    problem = _make_problem(3)
    fast = FAST(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(ValueError):
        FASTDesign(M=4).sampleWithMeta(problem, 10, seed=1)  # too small

    # must be strictly greater than 4*M^2 for stable frequency allocation
    X, meta = FASTDesign(M=4).sampleWithMeta(problem, 65, seed=123)
    res = fast.analyze(problem, X, Y=None, meta=meta, target="objs", index="all")
    metricNames = {metric.name for metric in res.metrics}
    assert "S1" in metricNames and "S1_norm" in metricNames
    assert "ST" in metricNames and "ST_norm" in metricNames


def test_rbd_fast_sample_and_analyze_smoke():
    problem = _make_problem(3)
    rbd = RBDFAST(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(ValueError):
        FASTDesign(M=4).sampleWithMeta(problem, 64, seed=1)  # must be > 4*M^2

    X, _ = LHS("classic").sampleWithMeta(problem, 65, seed=123)
    res = rbd.analyze(problem, X, Y=None, target="objs", index="all")
    metricNames = {metric.name for metric in res.metrics}
    assert "S1" in metricNames


def test_rsa_sample_and_analyze_smoke():
    problem = _make_problem(3)
    rsa = RSA(nRegion=4, verboseFlag=False, logFlag=False, saveFlag=False)

    X = LHS("classic").sample(problem, 60, seed=123)
    res = rsa.analyze(problem, X, Y=None, target="objs", index="all")
    metricNames = {metric.name for metric in res.metrics}
    assert "S1" in metricNames and "S1_norm" in metricNames


def test_rbd_fast_multi_output_uses_current_output_only():
    @ProblemABC.singleFunc
    def obj_func(x):
        x = np.asarray(x)
        return np.array([x[0], 2.0 * x[1]])

    problem = Problem(nInput=3, nObj=2, ub=1.0, lb=0.0, objFunc=obj_func, optType="min")
    rbd = RBDFAST(verboseFlag=False, logFlag=False, saveFlag=False)

    X = LHS("classic").sample(problem, 65, seed=123)
    Y = np.column_stack([X[:, 0], 2.0 * X[:, 1]])
    res = rbd.analyze(problem, X, Y=Y, target="objs", index="all")

    s1Metric = next(metric for metric in res.metrics if metric.name == "S1")
    assert s1Metric.values.shape == (2, problem.nInput)
    assert not np.allclose(s1Metric.values[0], s1Metric.values[1])


def test_fast_constant_output_returns_finite_zero_indices():
    problem = _make_problem(3)
    fast = FAST(verboseFlag=False, logFlag=False, saveFlag=False)
    X, meta = FASTDesign(M=4).sampleWithMeta(problem, 65, seed=123)
    Y = np.ones((X.shape[0], 1))

    res = fast.analyze(problem, X, Y=Y, meta=meta, target="objs", index="all")
    for metric in res.metrics:
        assert np.all(np.isfinite(metric.values))
        assert np.allclose(metric.values, 0.0)


def test_fast_compute_orders_matches_salib_reference_formula():
    outputs = np.sin(np.linspace(0.0, 4.0 * np.pi, 65, endpoint=False))
    n = outputs.size
    M = 4
    omega = (n - 1) // (2 * M)

    s1, st = FAST._computeOrders(outputs, n, M, omega)

    f = np.fft.fft(outputs)
    sp = np.power(np.absolute(f[np.arange(1, int(np.ceil(n / 2)))]) / n, 2)
    v = 2.0 * np.sum(sp)
    d1 = 2.0 * np.sum(sp[np.arange(1, M + 1, dtype=np.int32) * omega - 1])
    dt = 2.0 * np.sum(sp[np.arange(int(np.floor(omega / 2.0)), dtype=np.int32)])

    assert np.isclose(s1, d1 / v)
    assert np.isclose(st, 1.0 - dt / v)
    assert np.isfinite(s1)
    assert np.isfinite(st)


def test_sobol_multi_output_columnwise_normalization():
    problem = _make_problem(3)
    sob = Sobol(verboseFlag=False, logFlag=False, saveFlag=False)
    X, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, 8, seed=123)
    Y = np.column_stack([X[:, 0], 1000.0 * X[:, 1]])

    res = sob.analyze(problem, X, Y=Y, meta=meta, target="objs", index="all")
    s1Metric = next(metric for metric in res.metrics if metric.name == "S1")
    assert s1Metric.values.shape == (2, problem.nInput)
    assert np.all(np.isfinite(s1Metric.values))


def test_rsa_sparse_bins_remain_finite():
    problem = _make_problem(3)
    rsa = RSA(nRegion=20, verboseFlag=False, logFlag=False, saveFlag=False)
    X = LHS("classic").sample(problem, 20, seed=123)
    Y = np.zeros((20, 1))
    Y[:5, 0] = 1.0

    res = rsa.analyze(problem, X, Y=Y, target="objs", index="all")
    for metric in res.metrics:
        assert np.all(np.isfinite(metric.values))


def test_rbd_fast_bias_correction_is_clipped_to_unit_interval(monkeypatch):
    problem = _make_problem(3)
    rbd = RBDFAST(M=4, verboseFlag=False, logFlag=False, saveFlag=False)

    X = LHS("classic").sample(problem, 65, seed=123)
    Y = np.column_stack([X[:, 0]])

    def fake_periodogram(_):
        pxx = np.zeros(10)
        pxx[1] = 0.01
        pxx[2] = 0.99
        return np.arange(10), pxx

    import UQPyL.analysis.methods.rbd_fast as rbd_fast_mod

    monkeypatch.setattr(rbd_fast_mod, "periodogram", fake_periodogram)

    res = rbd.analyze(problem, X, Y=Y, target="objs", index="all")
    s1Metric = next(metric for metric in res.metrics if metric.name == "S1")
    assert np.all(s1Metric.values >= 0.0)
    assert np.all(s1Metric.values <= 1.0)


def test_mars_sample_and_analyze_smoke():
    problem = _make_problem(3)
    if MARS is None:
        pytest.skip("MARS extension modules are not available in this environment.")
    mars = MARS(verboseFlag=False, logFlag=False, saveFlag=False)

    X = LHS("classic").sample(problem, 40, seed=123)
    res = mars.analyze(problem, X, Y=None, target="objs", index="all")
    metricNames = {metric.name for metric in res.metrics}
    assert "S1" in metricNames and "S1_norm" in metricNames


def test_analysis_log_writes_full_metric_table():
    workDir = Path(".cache") / "analysis_log_test" / uuid.uuid4().hex
    workDir.mkdir(parents=True, exist_ok=True)
    try:
        problem = _make_problem(3)
        problem.workDir = str(workDir)
        fast = FAST(verboseFlag=False, logFlag=True, saveFlag=False)

        X, meta = FASTDesign(M=4).sampleWithMeta(problem, 65, seed=123)
        fast.analyze(problem, X, Y=None, meta=meta, target="objs", index="all")

        logFiles = list((workDir / "Result").glob("*.log"))
        assert logFiles
        text = logFiles[0].read_text(encoding="utf-8")
        assert "[S1]" in text
        assert "columns:" in text
        assert "obj1:" in text
        assert "target: objs" in text
        assert "meta:" in text
    finally:
        shutil.rmtree(workDir, ignore_errors=True)


def test_ana_result_convenience_api():
    problem = _make_problem(3)
    fast = FAST(verboseFlag=False, logFlag=False, saveFlag=False)
    X, meta = FASTDesign(M=4).sampleWithMeta(problem, 65, seed=123)

    res = fast.analyze(problem, X, Y=None, meta=meta, target="objs", index="all")

    assert res.problemName == problem.name
    assert res.runId is None
    assert "S1" in res.metricNames
    assert res.metricMap["S1"] is res.getMetric("S1")
    assert res["S1"] is res.getMetric("S1")

    summary = res.summary()
    assert summary["method"] == "FAST"
    assert "S1" in summary["metric_names"]

    payload = res.toDict()
    assert payload["method"] == "FAST"
    assert payload["run_id"] is None
    assert payload["meta"]["designType"] == "fast"
    assert isinstance(payload["metrics"], list)
    assert payload["metrics"][0]["name"] in res.metricNames


def test_ana_reader_context_and_metric_lookup(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE run (
            runId TEXT PRIMARY KEY,
            method TEXT NOT NULL,
            problem TEXT NOT NULL,
            target TEXT,
            nInput INTEGER NOT NULL,
            nOutput INTEGER NOT NULL,
            nCon INTEGER NOT NULL,
            status TEXT NOT NULL,
            runtime REAL,
            createdAt TEXT NOT NULL,
            finishedAt TEXT,
            problemPayload BLOB
        );

        CREATE TABLE runParam (
            runId TEXT NOT NULL,
            name TEXT NOT NULL,
            value TEXT
        );

        CREATE TABLE metric (
            metricId INTEGER PRIMARY KEY AUTOINCREMENT,
            runId TEXT NOT NULL,
            name TEXT NOT NULL,
            rowLabels TEXT,
            colLabels TEXT,
            valueJson TEXT,
            colDim TEXT NOT NULL
        );

        CREATE TABLE artifact (
            artifactId INTEGER PRIMARY KEY AUTOINCREMENT,
            runId TEXT NOT NULL,
            name TEXT NOT NULL,
            payload BLOB
        );
        """
    )
    conn.execute(
        """
        INSERT INTO run (
            runId, method, problem, target, nInput, nOutput, nCon,
            status, runtime, createdAt, finishedAt, problemPayload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "demo_001",
            "FAST",
            "DemoProblem",
            "objs",
            3,
            1,
            0,
            "finished",
            0.123,
            "2026-05-03T12:00:00",
            "2026-05-03T12:00:01",
            sqlite3.Binary(pickle.dumps({"problem": "demo"})),
        ),
    )
    conn.execute(
        "INSERT INTO metric (runId, name, rowLabels, colLabels, valueJson, colDim) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "demo_001",
            "S1",
            json.dumps(["obj1"]),
            json.dumps(["x1", "x2", "x3"]),
            json.dumps([[0.5, 0.3, 0.2]]),
            "input",
        ),
    )
    conn.execute(
        "INSERT INTO artifact (runId, name, payload) VALUES (?, ?, ?)",
        ("demo_001", "settings", sqlite3.Binary(pickle.dumps({"M": 4}))),
    )
    conn.execute(
        "INSERT INTO artifact (runId, name, payload) VALUES (?, ?, ?)",
        ("demo_001", "extra", sqlite3.Binary(pickle.dumps({"note": "demo"}))),
    )
    conn.commit()

    def _connect(_):
        return conn

    monkeypatch.setattr("UQPyL.analysis.runtime.reader.sqlite3.connect", _connect)

    with AnaReader("dummy.sqlite3") as reader:
        metric = reader.get_metric("S1")
        res = reader.load_result()
        summary = reader.get_run_summary()

    assert metric.name == "S1"
    assert res.runId == "demo_001"
    assert res.problemName == "DemoProblem"
    assert summary["run_id"] == "demo_001"
    assert summary["problem_name"] == "DemoProblem"
    assert "S1" in summary["metric_names"]
    assert "settings" in summary["artifact_names"]
    assert np.allclose(metric.values, res.getMetric("S1").values)
    assert res.settings["M"] == 4
    assert res.target == "objs"
    assert res.extra["note"] == "demo"


def test_analysis_storage_runid_includes_problem_slug():
    problem = _make_problem(3)
    problem.name = "My Problem#1"
    fast = FAST(verboseFlag=False, logFlag=False, saveFlag=False)
    fast.setProblem(problem)
    storage = __import__("UQPyL.analysis.runtime.storage", fromlist=["SqliteStorage"]).SqliteStorage(".cache")

    dbPath, runId = storage._db_path(fast.name, problem.name)

    assert "fast_My_Problem_1_" in runId
    assert dbPath.endswith(f"{runId}.sqlite3")

    commonDbPath, commonRunId = build_db_path(".cache/Result", fast.name, problem.name)
    assert "fast_My_Problem_1_" in commonRunId
    assert commonDbPath.endswith(f"{commonRunId}.sqlite3")


def test_analysis_list_runs_includes_filename():
    dbPath = Path(r"D:\UQPyL\.cache\analysis_list_runs\fast_Demo_20260503_1817_abcd.sqlite3")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE run (
            runId TEXT PRIMARY KEY,
            method TEXT NOT NULL,
            problem TEXT NOT NULL,
            target TEXT,
            status TEXT NOT NULL,
            runtime REAL,
            createdAt TEXT NOT NULL,
            finishedAt TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO run (
            runId, method, problem, target, status, runtime, createdAt, finishedAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "fast_Demo_20260503_1817_abcd",
            "FAST",
            "Demo",
            "objs",
            "finished",
            0.1,
            "2026-05-03T18:17:00",
            "2026-05-03T18:17:01",
        ),
    )
    conn.commit()

    origConnect = sqlite3.connect
    origGlob = Path.glob

    def fake_connect(path, *args, **kwargs):
        if str(path) == str(dbPath):
            return conn
        return origConnect(path, *args, **kwargs)

    def fake_glob(self, pattern):
        if pattern == "*.sqlite3":
            return [dbPath]
        return list(origGlob(self, pattern))

    import UQPyL.analysis.runtime.reader as reader_mod

    reader_mod.sqlite3.connect = fake_connect
    Path.glob = fake_glob
    try:
        runs = AnaReader.list_runs(".cache")
        assert len(runs) >= 1
        matched = next(item for item in runs if item["run_id"] == "fast_Demo_20260503_1817_abcd")
        assert matched["fileName"] == dbPath.name
        assert matched["dbPath"].endswith(dbPath.name)
    finally:
        reader_mod.sqlite3.connect = origConnect
        Path.glob = origGlob
        conn.close()




