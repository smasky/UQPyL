from pathlib import Path

import numpy as np
import pytest

from UQPyL.optimization import OptHistory, OptReader, OptResult, Result, SqliteStorage, Verbose
from UQPyL.optimization.core import NDSort, crowdingDist, gaOperator, tourSelect, uniformPoint
from UQPyL.optimization.base import AlgorithmABC
from UQPyL.analysis.base import AnalysisABC
from UQPyL.inference.base import InferenceABC
from UQPyL.core.params import Params
from UQPyL.core.runtime_session import RunSession
from UQPyL.surrogate.setting import Setting


def test_core_exports_available_after_layout_refactor():
    assert callable(NDSort)
    assert callable(crowdingDist)
    assert callable(gaOperator)
    assert callable(tourSelect)
    assert callable(uniformPoint)


def test_runtime_exports_available_after_layout_refactor():
    assert AlgorithmABC is not None
    assert OptReader is not None
    assert OptHistory is not None
    assert OptResult is not None
    assert Result is not None
    assert SqliteStorage is not None
    assert Verbose is not None


def test_algorithm_base_accepts_hv_reference_point():
    class _DummyAlg(AlgorithmABC):
        def run(self, problem, seed=None):
            return None

    alg = _DummyAlg(verboseFlag=False, logFlag=False, saveFlag=False, hvRefPoint=[1.5, 2.5])
    assert alg.hvRefPoint is not None
    assert alg.get('hvRefPoint').tolist() == [1.5, 2.5]


def test_core_modules_no_longer_forward_to_optimization_util():
    core_dir = Path(__file__).resolve().parents[1] / "UQPyL" / "optimization" / "core"
    for path in core_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert ".util" not in text


def test_shared_params_class_is_used_by_runtime_bases():
    class _DummyAnalysis(AnalysisABC):
        name = "DummyAnalysis"

        def _analyzeCore(self, problem, *args, **kwargs):
            return None

    class _DummyAlgorithm(AlgorithmABC):
        def run(self, problem, seed=None):
            return None

    class _DummyInference(InferenceABC):
        def run(self, problem=None, *args, **kwargs):
            return None

    with pytest.raises(TypeError):
        AlgorithmABC(verboseFlag=False, logFlag=False, saveFlag=False)

    with pytest.raises(TypeError):
        InferenceABC(verboseFlag=False, logFlag=False, saveFlag=False)

    assert isinstance(_DummyAlgorithm(verboseFlag=False, logFlag=False, saveFlag=False).params, Params)
    assert isinstance(_DummyInference(verboseFlag=False, logFlag=False, saveFlag=False).params, Params)

    analysis = _DummyAnalysis(verboseFlag=False, logFlag=False, saveFlag=False)
    assert isinstance(analysis.setting, Params)


def test_runtime_bases_use_session_without_duplicate_storage_context(monkeypatch):
    class _Problem:
        name = "Demo"
        nInput = 2
        nObj = 1
        nOutput = 1
        nCon = 0
        nCons = 0
        opt = 1

        @staticmethod
        def apply_var_type(decs):
            return decs

    class _Storage:
        def __init__(self, root_dir):
            self.root_dir = root_dir

        def create_run(self, obj):
            return RunSession(run_id="demo_run", db_path="Result/demo.sqlite3", root_dir=self.root_dir)

    class _DummyAnalysis(AnalysisABC):
        name = "DummyAnalysis"

        def _analyzeCore(self, problem, *args, **kwargs):
            return None

    class _DummyAlgorithm(AlgorithmABC):
        name = "DummyAlgorithm"

        def run(self, problem, seed=None):
            return None

    class _DummyInference(InferenceABC):
        name = "DummyInference"

        def run(self, problem=None, *args, **kwargs):
            return None

    monkeypatch.setattr("UQPyL.analysis.base.SqliteStorage", _Storage)
    monkeypatch.setattr("UQPyL.optimization.base.SqliteStorage", _Storage)
    monkeypatch.setattr("UQPyL.inference.base.SqliteStorage", _Storage)

    problem = _Problem()

    analysis = _DummyAnalysis(verboseFlag=False, logFlag=False, saveFlag=True)
    analysis.setup(problem)
    assert analysis.session.run_id == "demo_run"
    assert not hasattr(analysis, "storageCtx")

    algorithm = _DummyAlgorithm(verboseFlag=False, logFlag=False, saveFlag=True)
    algorithm.setup(problem, seed=123)
    assert algorithm.session.run_id == "demo_run"
    assert not hasattr(algorithm, "storageCtx")

    inference = _DummyInference(verboseFlag=False, logFlag=False, saveFlag=True)
    inference.setup(problem, seed=123)
    assert inference.session.run_id == "demo_run"
    assert not hasattr(inference, "storageCtx")


def test_surrogate_setting_provides_params_like_base_interface():
    setting = Setting()
    setting.set("alpha", 1.0)
    setting.set("beta", 2.0)

    assert tuple(setting.keys()) == ("alpha", "beta")
    assert tuple(setting.values()) == (1.0, 2.0)
    assert tuple(setting.items()) == (("alpha", 1.0), ("beta", 2.0))
    assert setting.asDict() == {"alpha": 1.0, "beta": 2.0}


def test_surrogate_setting_keeps_base_mapping_in_sync_after_mutations():
    setting = Setting()
    setting.set("alpha", 1.0, attr={"lb": 0.0, "ub": 2.0, "type": "float"})
    setting.set("beta", 2.0)

    paraInfos, _, _ = setting.getParaInfos(["alpha"])
    setting.setVals(paraInfos, np.array([1.5]))
    assert setting.asDict()["alpha"] == 1.5

    other = Setting()
    other.set("gamma", 3.0)
    setting.mergeSetting(other)
    assert setting.asDict()["gamma"] == 3.0

    setting.removeParas(["beta"])
    assert "beta" not in setting.asDict()
    assert setting.dicts["gamma"] == 3.0
