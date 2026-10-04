"""Run the actual maintained Markdown workflows, including in wheel tests."""

import contextlib
import io
from pathlib import Path
import pickle
import re

import numpy as np
import pytest


DOCS = Path(__file__).resolve().parents[1] / "docs_v2"


def pythonBlocks(page):
    return re.findall(r"```python\s*\n(.*?)```", (DOCS / page).read_text(encoding="utf-8"), re.S)


def executeBlock(code, label, namespace=None):
    namespace = {"__name__": "documented_workflow"} if namespace is None else namespace
    with np.printoptions(), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, label, "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "page,count", [("examples.md", 9), ("cn/examples.md", 7), ("quick_start.md", 10), ("cn/quick_start.md", 10)]
)
def testDocumentedCompleteWorkflows(page, count, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = pythonBlocks(page)
    assert len(blocks) == count
    state = pickle.dumps(np.random.get_state())
    namespace = {"__name__": "documented_workflow"}
    for index, code in enumerate(blocks):
        namespace = executeBlock(code, f"{page}:block{index + 1}", None if "examples" in page else namespace)
        if "res = problem.evaluate([[1.0, 2.2]])" in code:
            np.testing.assert_allclose(namespace["res"].objs, [[0.02]])
            np.testing.assert_allclose(namespace["res"].sims, [[1.0, 2.2]])
        if "RandSelect(" in code:
            replay = executeBlock(code, page)
            np.testing.assert_array_equal(replay["testIdx"], namespace["testIdx"])
            np.testing.assert_allclose(replay["pred"], namespace["pred"])
        if "checkedObj =" in code:
            np.testing.assert_allclose(
                namespace["checkedObj"], namespace["expensiveProblem"].evaluate(namespace["result"].bestDecs).objs
            )
    assert pickle.dumps(np.random.get_state()) == state


@pytest.mark.parametrize("page", ["surrogate.md", "cn/surrogate.md"])
def testDocumentedTuningIsReproducible(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selected = [code for code in pythonBlocks(page) if "tuner.gridTune(" in code]
    assert len(selected) == 1
    first, second = [executeBlock(selected[0], page) for _ in range(2)]
    np.testing.assert_array_equal(first["tuner"].lastSplit["train_indices"], second["tuner"].lastSplit["train_indices"])
    assert first["bestScore"] == second["bestScore"]
    np.testing.assert_allclose(first["model"].predict([[0.5]]), [[0.35]], atol=0.001)


def testDocumentedModelEvaluatorAndEvalContract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selected = [code for code in pythonBlocks("api/problem.md") if "class MSEEvaluator" in code]
    assert len(selected) == 1
    namespace = executeBlock(selected[0], "api/problem.md:MSEEvaluator")
    problem = namespace["problem"]
    np.testing.assert_allclose(problem.evaluate([[1.0, 2.2]]).objs, [[0.02]])
    onlySims = problem.evaluate([[1.0, 2.2]], target="sims")
    assert onlySims.objs is None and onlySims.cons is None
    np.testing.assert_allclose(onlySims.sims, [[1.0, 2.2]])


@pytest.mark.parametrize("page,expectedCount", [("problem.md", 3), ("cn/problem.md", 2)])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("maskKind", ["none", "partial", "all"])
def testDocumentedMaskedObjectives(page, expectedCount, flat, maskKind):
    import ast
    from UQPyL.problem import Eval, SimContext

    obs = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = None if maskKind == "none" else np.array([[False, True], [False, False]])
    if maskKind == "all":
        mask[:] = True
    values = np.array([[2.0, 4.0, 5.0, 8.0], [3.0, 5.0, 6.0, 9.0]])
    if mask is not None:
        values[:, mask.reshape(-1)] = np.nan
    context = SimContext(
        values if flat else values.reshape(2, 2, 2).reshape(2, 4),
        obs.reshape(-1),
        None if mask is None else mask.reshape(-1),
    )
    count = 0
    for code in pythonBlocks(page):
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, ast.FunctionDef) or node.name not in {"objFunc", "evaluate"}:
                continue
            if "simContext.mask" not in ast.unparse(node):
                continue
            count += 1
            namespace = {"np": np, "Eval": Eval}
            exec(compile(ast.Module(body=[node], type_ignores=[]), page, "exec"), namespace)
            function = namespace[node.name]
            args = (None, np.ones((2, 1)), context) if node.name == "evaluate" else (np.ones((2, 1)), context)
            if maskKind == "all":
                with pytest.raises(ValueError, match="No valid observations"):
                    function(*args)
            else:
                result = function(*args)
                actual = result.objs if isinstance(result, Eval) else result
                valid = np.ones(4, dtype=bool) if mask is None else ~mask.reshape(-1)
                expected = np.mean((values[:, valid] - obs.reshape(-1)[valid]) ** 2, axis=1, keepdims=True)
                np.testing.assert_array_equal(actual, expected)
    assert count == expectedCount


@pytest.mark.parametrize("page", ["calibration.md", "cn/calibration.md", "api/calibration.md"])
def testDocumentedCalibrationUsesVectorObservations(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selected = [
        code for code in pythonBlocks(page) if "problem = ModelProblem(" in code and "import numpy as np" in code
    ]
    assert selected
    for code in selected:
        namespace = executeBlock(code, page)
        problem = namespace["problem"]
        assert problem.obs.ndim == 1
        assert problem.mask is None or problem.mask.shape == problem.obs.shape
        sims = problem.evaluate(np.zeros((1, problem.nInput)), target="sims").sims
        assert sims.shape == (1, problem.nObs)
        if problem.mask is not None:
            np.testing.assert_array_equal(problem.mask, [False, True, False, True])
            np.testing.assert_array_equal(sims, [[0.0, 999.0, 0.0, 999.0]])
