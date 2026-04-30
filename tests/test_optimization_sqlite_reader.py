from pathlib import Path

from UQPyL.optimization.runtime import OptReader
from UQPyL.optimization.soea import GA
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

    runs = OptReader.listRuns(result_dir)
    assert len(runs) >= 1

    reader = OptReader(str(new_files[0]))
    run = reader.getRun()
    params = reader.getRunParams()
    snapshots = reader.listSnapshots()
    pop = reader.loadLastPopulation()
    best = reader.loadLastBest()
    alg2 = reader.loadAlgorithm()
    prob2 = reader.loadProblem()
    reader.close()

    assert run["algorithm"] == "GA"
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
