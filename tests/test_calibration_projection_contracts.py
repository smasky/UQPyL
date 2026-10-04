"""Calibration projection contracts.

Migrated from test_review_c01_c05.py; original regression provenance is retained below.
"""

import numpy as np
import pytest
from UQPyL.calibration import ES, IES
from UQPyL.problem import Problem, ModelProblem


# Regression source: test_review_c01_c05.py::testEnsembleProjectsBeforeSimulation
@pytest.mark.parametrize("methodClass", [ES, IES])
@pytest.mark.parametrize("target,expected", [(10.0, 1.0), (-10.0, 0.0), (0.5, 0.5)])
def testEnsembleProjectsBeforeSimulation(methodClass, target, expected):
    seen = []

    def simulate(x):
        assert np.all((x >= 0) & (x <= 1))
        seen.append(x.copy())
        return x

    problem = ModelProblem(nInput=1, lb=0.0, ub=1.0, simFunc=simulate, obs=np.array([target]))
    method = methodClass(**({"maxIters": 3} if methodClass is IES else {}))
    x = np.array([[0.2], [0.8]])
    result = method.run(problem, x)
    np.testing.assert_allclose(result.posteriorDecs, expected)
    np.testing.assert_allclose(result.posteriorSims, expected)
    np.testing.assert_array_equal(x, [[0.2], [0.8]])
    assert len(seen) == (4 if methodClass is IES else 2)
    assert result.diagnostics["boundUpdates"][0]["adjusted_members"] == (0 if target == 0.5 else 2)


# Regression source: test_review_c01_c05.py::testEnsembleRejectsUnsupportedInitialDomainBeforeSimulation
@pytest.mark.parametrize("methodClass", [ES, IES])
@pytest.mark.parametrize("kind", ["outside", "nan", "integer", "discrete", "constraint", "reversed"])
def testEnsembleRejectsUnsupportedInitialDomainBeforeSimulation(methodClass, kind):
    seen = []

    def simulate(x):
        seen.append(x.copy())
        return x

    options = {}
    if kind == "integer":
        options = dict(varType=[1])
    elif kind == "discrete":
        options = dict(varType=[2], varSet={0: [0.2, 0.8]})
    elif kind == "constraint":
        options = dict(nCon=1, objFunc=lambda x, ctx: x, conFunc=lambda x, ctx: x - 0.5)
    problem = ModelProblem(nInput=1, lb=0.0, ub=1.0, simFunc=simulate, obs=np.array([0.5]), **options)
    x = np.array([[0.2], [0.8]])
    if kind == "outside":
        x[0] = -1
    elif kind == "nan":
        x[0] = np.nan
    elif kind == "reversed":
        problem.lb[:] = 2
    with pytest.raises((ValueError, NotImplementedError)):
        methodClass().run(problem, x)
    assert seen == []


# Regression source: test_review_c01_c05.py::testEnsembleFixedBoundAndInteriorUpdate
@pytest.mark.parametrize("methodClass", [ES, IES])
def testEnsembleFixedBoundAndInteriorUpdate(methodClass):
    x = np.array([[0.2, 2.0], [0.8, 2.0]])
    problem = ModelProblem(nInput=2, lb=[0, 2], ub=[1, 2], simFunc=lambda x: x, obs=np.array([0.5, 8.0]))
    result = methodClass().run(problem, x)
    np.testing.assert_allclose(result.posteriorDecs, [[0.5, 2.0], [0.5, 2.0]])


# Regression source: test_review_c01_c05.py::testProjectedCalibrationResultSurvivesSqlite
@pytest.mark.parametrize("methodClass", [ES, IES])
def testProjectedCalibrationResultSurvivesSqlite(tmp_path, methodClass):
    from UQPyL.calibration import CalReader

    problem = ModelProblem(nInput=1, lb=0.0, ub=1.0, simFunc=lambda x: x, obs=np.array([10.0]))
    problem.workDir = str(tmp_path)
    result = methodClass(saveFlag=True).run(problem, np.array([[0.2], [0.8]]))
    with CalReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        loaded = reader.load_result()
    np.testing.assert_array_equal(loaded.posteriorDecs, result.posteriorDecs)
    np.testing.assert_array_equal(loaded.posteriorSims, result.posteriorSims)
    assert loaded.diagnostics["boundUpdates"] == result.diagnostics["boundUpdates"]
