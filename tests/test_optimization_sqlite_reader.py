from pathlib import Path

from UQPyL.optimization.runtime import OptReader
from UQPyL.optimization.runtime.storage import SqliteStorage
from UQPyL.optimization.soea import GA
from UQPyL.core.runtime import build_db_path, make_run_id
from UQPyL.problem.sop.single_simple_problem import Sphere


def test_sqlite_save_and_reader_roundtrip():
    result_dir = Path("Result")
    before = set(result_dir.glob("*.sqlite3")) if result_dir.exists() else set()
    problem = Sphere(nInput=3, ub=1.0, lb=-1.0)

    alg = GA(
        nPop=6,
        maxFEs=18,
        maxIters=3,
        tolerate=None,
        verboseFlag=False,
        logFlag=False,
        saveFlag=True,
        saveFreq=2,
    )
    res = alg.run(problem, seed=123)
    assert res.bestObjs.shape == (1, 1)

    after = set(result_dir.glob("*.sqlite3"))
    new_files = sorted(after - before)
    assert len(new_files) == 1

    runs = OptReader.list_runs(result_dir)
    assert len(runs) >= 1

    reader = OptReader(str(new_files[0]))
    run = reader.get_run()
    summary = reader.get_run_summary()
    params = reader.get_run_params()
    snapshots = reader.list_snapshots()
    pop = reader.load_last_population()
    best = reader.load_last_best()
    alg2 = reader.load_algorithm()
    prob2 = reader.load_problem()
    reader.close()

    assert run["algorithm"] == "GA"
    assert summary["method"] == "GA"
    assert summary["problem_name"] == "Sphere"
    assert "seed" in params
    assert len(snapshots) >= 1
    assert pop.decs.shape[1] == 3
    assert best.objs.shape == (1, 1)
    assert alg2.name == "GA"
    assert prob2.name == "Sphere"


def test_log_file_contains_full_summary_and_final():
    result_dir = Path("Result")
    before = set(result_dir.glob("*.log")) if result_dir.exists() else set()
    problem = Sphere(nInput=3, ub=1.0, lb=-1.0)

    alg = GA(
        nPop=6,
        maxFEs=18,
        maxIters=3,
        tolerate=None,
        verboseFlag=False,
        verboseFreq=1,
        logFlag=True,
        saveFlag=False,
    )
    alg.run(problem, seed=123)

    after = set(result_dir.glob("*.log"))
    new_logs = sorted(after - before)
    assert len(new_logs) == 1
    text = new_logs[0].read_text()
    assert "[summary] GA" in text
    assert "best X" in text
    assert "Optimization finished" in text


def test_optimization_storage_runid_includes_problem_slug():
    storage = SqliteStorage("Result")
    dbPath, runId = storage._db_path("GA", "My Problem#1")

    assert "ga_My_Problem_1_" in runId
    assert dbPath.endswith(f"{runId}.sqlite3")


def test_runtime_common_helpers_match_storage_naming():
    runId = make_run_id("GA", "My Problem#1")
    assert runId.startswith("ga_My_Problem_1_")

    dbPath, explicitRunId = build_db_path("Result", "GA", "My Problem#1")
    assert explicitRunId.startswith("ga_My_Problem_1_")
    assert dbPath.endswith(f"{explicitRunId}.sqlite3")
