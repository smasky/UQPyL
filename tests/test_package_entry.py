import importlib
import sys

import pytest


def test_package_entry_re_raises_non_module_not_found(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "UQPyL.surrogate":
            raise RuntimeError("boom")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    sys.modules.pop("UQPyL", None)

    with pytest.raises(RuntimeError, match="boom"):
        importlib.import_module("UQPyL")
