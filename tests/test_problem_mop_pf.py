import numpy as np
import pytest

from UQPyL.problem.mop import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6


def test_zdt_pf_methods_and_error_branches():
    # error branches: bi-objective only
    for cls in [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]:
        with pytest.raises(ValueError):
            cls(nInput=10, nOutput=3)

    assert len(ZDT1().getPF()) == 2
    assert len(ZDT2().getPF()) == 2
    assert len(ZDT3().getPF()) == 2
    # ZDT4 uses `GetPF` (capital G)
    assert len(ZDT4().GetPF()) == 2
    assert len(ZDT6().getPF()) == 2


def test_dtlz_pf_methods_and_error_branches():
    # DTLZ2/4/5/6/7 enforce 3 objectives in this implementation
    for cls in [DTLZ2, DTLZ4, DTLZ5, DTLZ6, DTLZ7]:
        with pytest.raises(ValueError):
            cls(nInput=7, nOutput=2)

    # Call getPF for all, which covers the large missing blocks.
    # Keep default nOutput=3.
    for cls in [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7]:
        p = cls(nInput=7, nOutput=3)
        pf = p.getPF()
        assert isinstance(pf, tuple)
        assert len(pf) == 3
        # PF arrays should be numeric and broadcastable
        for arr in pf:
            a = np.asarray(arr)
            assert a.size > 0


