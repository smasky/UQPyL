import os
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from UQPyL.util.verbose import Verbose, save_dict_to_hdf5


class _DummyEmitter:
    def __init__(self):
        self.sent = []

    def send(self, text=None):
        self.sent.append(text)


class _DummyResult:
    def __init__(self):
        self.runtime = None

    def generateNetCDF(self):
        # small, valid xarray Dataset (we won't write it to disk in this test)
        return xr.Dataset({"x": (("n",), np.array([1.0, 2.0]))})


def test_verbose_format_time_and_table_helpers():
    s = Verbose.formatTime(3661.5)
    assert "hour" in s and "minute" in s

    tables = Verbose.verboseTable(["a", "b", "c", "d"], ["1", "2", "3", "4"], num=2, width=40)
    assert len(tables) == 2


def test_verbose_output_appends_logs(monkeypatch):
    # Avoid printing even if verboseFlag is True
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    problem = SimpleNamespace(verboseFlag=False, logLines=[])
    from prettytable import PrettyTable

    t = PrettyTable(["A"])
    t.add_row(["1"])
    Verbose.output(t, problem)
    assert len(problem.logLines) >= 1


def test_verbose_run_decorator_smoke_no_io(monkeypatch, tmp_path):
    # Ensure checkDir doesn't touch real workspace
    monkeypatch.setattr(Verbose, "workDir", str(tmp_path), raising=False)
    monkeypatch.setattr(Verbose, "checkDir", lambda wd: (str(tmp_path), str(tmp_path)), raising=True)
    monkeypatch.setattr(Verbose, "saveData", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(Verbose, "saveLog", lambda *a, **k: None, raising=True)

    problem = SimpleNamespace(
        name="P",
        nInput=1,
        nOutput=1,
        optType="min",
        xLabels=["x1"],
        yLabels=["y1"],
        verboseFlag=False,
        logLines=None,
    )

    obj = SimpleNamespace(
        name="ALG",
        problem=problem,
        verboseFlag=False,
        verboseFreq=1,
        logFlag=False,
        saveFlag=False,
        setting=SimpleNamespace(keys=["k"], values=[1]),
        result=_DummyResult(),
        FEs=0,
        iters=0,
    )

    class _Res:
        bestDecs_True = np.array([[0.0]])
        bestObjs_True = np.array([[0.0]])
        bestFeasible = True
        bestMetric = 0.0
        appearFEs = 0
        appearIters = 0

    @Verbose.run
    def _run(o, prob):
        return _Res()

    res_nc = _run(obj, problem)
    assert isinstance(res_nc, xr.Dataset)


def test_verbose_inference_decorator_smoke(monkeypatch, tmp_path):
    # inference() always calls checkDir/saveData in current implementation; stub them.
    monkeypatch.setattr(Verbose, "workDir", str(tmp_path), raising=False)
    monkeypatch.setattr(Verbose, "checkDir", lambda wd: (str(tmp_path), str(tmp_path)), raising=True)
    monkeypatch.setattr(Verbose, "saveData", lambda *a, **k: None, raising=True)

    problem = SimpleNamespace(
        name="P",
        nInput=1,
        nOutput=1,
        xLabels=["x1"],
        verboseFlag=False,
        logLines=None,
    )
    obj = SimpleNamespace(
        name="INF",
        problem=problem,
        verboseFlag=False,
        logFlag=False,
        saveFlag=False,
        setting=SimpleNamespace(keys=["k"], values=[1]),
    )

    @Verbose.inference
    def _infer(o, prob):
        return {"iter": 1, "bestDecs": np.array([[0.0]]), "bestObjs": np.array([[0.0]])}

    res = _infer(obj, problem)
    assert res["iter"] == 1


def test_save_dict_to_hdf5_nested(tmp_path):
    # HDF5 support was removed; keep a small guard test for the old API.
    with pytest.raises(NotImplementedError):
        save_dict_to_hdf5(None, {"a": 1, "b": {"c": np.array([1, 2, 3])}})

