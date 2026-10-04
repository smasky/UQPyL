"""Exercise wheel coverage reporting and required-import failures locally."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET

repoRoot = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wheelCheck", repoRoot / ".github/scripts/test_wheel.py")
wheelCheck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheelCheck)


def verifyReports():
    with tempfile.TemporaryDirectory() as directory:
        reportDir = Path(directory)
        coveragePath = reportDir / "coverage.xml"
        jobSummary = reportDir / "job-summary.md"
        jobSummary.write_text("Previous step\n")
        for filename in (
            "venv/lib/python3.14/site-packages/UQPyL/core/runtime.py",
            r"C:\venv\Lib\site-packages\UQPyL\core\runtime.py",
            "core/runtime.py",
            "UQPyL/core/runtime.py",
        ):
            root = ET.Element(
                "coverage", {"lines-covered": "8", "lines-valid": "10", "branches-covered": "3", "branches-valid": "4"}
            )
            ET.SubElement(ET.SubElement(root, "sources"), "source").text = "/temporary/venv"
            ET.SubElement(root, "class", {"filename": filename})
            ET.ElementTree(root).write(coveragePath)
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(jobSummary), "GITHUB_SHA": "test-sha"}):
                wheelCheck.writeCoverageSummary(coveragePath, reportDir, repoRoot)
            data = json.loads((reportDir / "coverage-summary.json").read_text())
            assert data["line_rate"] == 0.8 and data["branch_rate"] == 0.75
            assert data["commit_sha"] == "test-sha"
            tree = ET.parse(coveragePath)
            assert tree.find(".//class").get("filename") == "UQPyL/core/runtime.py"
            assert tree.find("./sources/source").text == str(repoRoot)
        assert jobSummary.read_text().startswith("Previous step\n")
        assert jobSummary.read_text().count("75.00%") == 4
        root.set("branches-valid", "0")
        ET.ElementTree(root).write(coveragePath)
        try:
            wheelCheck.writeCoverageSummary(coveragePath, reportDir, repoRoot)
        except ValueError:
            pass
        else:
            raise AssertionError("A line-only report was accepted as branch coverage")
    print(
        "Coverage reporting: four path formats, exact rates, append-only summary and missing-branch rejection passed."
    )


def verifyMissingImports():
    cases = [
        ("UQPyL.surrogate.mars.core._types", "tests/test_surrogate_mars_smoke.py"),
        ("UQPyL.surrogate.svr.core.libsvm_interface", "tests/test_surrogate_svr_smoke.py"),
        ("UQPyL.surrogate.regression.lasso.lasso", "tests/test_lasso_input_isolation.py"),
    ]
    results = []
    for module, testFile in cases:
        code = """
import importlib.abc
import sys
class MissingNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == sys.argv[1]:
            raise ModuleNotFoundError('injected missing native extension: ' + fullname, name=fullname)
sys.meta_path.insert(0, MissingNative())
import pytest
raise SystemExit(pytest.main(['--collect-only', '-q', '-W', 'error', sys.argv[2]]))
"""
        run = subprocess.run(
            [sys.executable, "-c", code, module, testFile], cwd=repoRoot, capture_output=True, text=True
        )
        assert run.returncode == 2, (module, run.stdout, run.stderr)
        assert "injected missing native extension" in run.stdout
        assert "skipped" not in run.stdout
        results.append({"module": module, "test_file": testFile, "exit_code": run.returncode, "skipped": False})
    Path(__file__).with_name("0929-required-import-failures.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Missing MARS, SVR and Lasso extensions: all three fail collection instead of skipping.")


if __name__ == "__main__":
    verifyReports()
    verifyMissingImports()
