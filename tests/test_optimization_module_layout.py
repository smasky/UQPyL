from pathlib import Path

from UQPyL.optimization import OptHistory, OptReader, OptResult, Result, SqliteStorage, Verbose
from UQPyL.optimization.core import NDSort, crowdingDist, gaOperator, tourSelect, uniformPoint
from UQPyL.optimization.base import AlgorithmABC


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
        pass

    alg = _DummyAlg(verboseFlag=False, logFlag=False, saveFlag=False, hvRefPoint=[1.5, 2.5])
    assert alg.hvRefPoint is not None
    assert alg.getParaVal('hvRefPoint').tolist() == [1.5, 2.5]


def test_core_modules_no_longer_forward_to_optimization_util():
    core_dir = Path(__file__).resolve().parents[1] / "UQPyL" / "optimization" / "core"
    for path in core_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert ".util" not in text
