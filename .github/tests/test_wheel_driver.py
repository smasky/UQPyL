"""The two wheel-test entry points must not accidentally install or build twice."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wheel_driver", ROOT / ".github/scripts/test_wheel.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.mark.parametrize("installed", [False, True])
def testDriverUsesOneInstallationAndAnExternalTestDirectory(tmp_path, monkeypatch, installed):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "placeholder.whl").touch()
    commands = []
    created = []
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("PYTHONHOME", "/wrong-home")
    monkeypatch.setattr(driver.venv.EnvBuilder, "create", lambda self, path: created.append(path))
    monkeypatch.setattr(driver.shutil, "copytree", lambda *args, **kwargs: None)

    def run(arguments, *, cwd, env, check):
        assert check
        assert Path(cwd) != ROOT
        assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
        commands.append(arguments)

    monkeypatch.setattr(driver.subprocess, "run", run)
    args = ["test_wheel.py", "--wheel-dir", str(wheels), "--report-dir", str(tmp_path / "reports")]
    if installed:
        args.append("--installed")
    monkeypatch.setattr(sys, "argv", args)
    driver.main()
    installs = [call for call in commands if call[1:4] == ["-m", "pip", "install"]]
    assert len(installs) == (0 if installed else 1)
    assert len(created) == (0 if installed else 1)
    assert any(call[1:4] == ["-m", "pip", "check"] for call in commands)
    assert any(call[1:3] == ["-I", "-c"] for call in commands)
    assert sum(call[1:3] == ["-m", "pytest"] for call in commands) == 1
    assert not any(call[1:3] == ["-m", "build"] for call in commands)
    if installed:
        assert all(call[0] == sys.executable for call in commands)


def testNoWheelFailsBeforeEnvironmentCreation(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_wheel.py", "--wheel-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
