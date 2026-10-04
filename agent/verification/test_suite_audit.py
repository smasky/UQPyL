"""Inventory test sources and an existing wheel report without running tests."""

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


repoRoot = Path(__file__).resolve().parents[2]
reportRoot = repoRoot / ".cache/py314-wheel-test"


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(name):
    name = name.removeprefix("test_")
    if name.startswith(("review_", "remaining_review")):
        return "historical_regressions"
    if name.startswith(("surrogate_", "autotuner_", "gpr_", "rbf_", "lasso_", "lbfgsb_", "mars_")):
        return "surrogate"
    if name.startswith(("optimization_", "hv_")):
        return "optimization"
    if name.startswith(("inference_", "dream_")):
        return "inference"
    if name.startswith("calibration_"):
        return "calibration"
    if name.startswith("analysis_"):
        return "analysis"
    if name.startswith(("problem", "model_problem")):
        return "problem"
    if name.startswith(("doe_", "lhs_")):
        return "doe"
    if name.startswith(("user_workflows", "documented_workflows")):
        return "workflows"
    return "runtime_util_viz_package"


def callName(node):
    return ast.unparse(node.func) if isinstance(node, ast.Call) else ""


def main():
    junitPath = reportRoot / "junit.xml"
    coveragePath = reportRoot / "coverage.xml"
    junit = ET.parse(junitPath).getroot()
    cases = junit.findall(".//testcase")
    caseCounts = Counter()
    categoryCounts = Counter()
    fileTimes = defaultdict(float)
    for case in cases:
        filePath = case.get("classname").replace(".", "/") + ".py"
        caseCounts[(filePath, case.get("name").split("[")[0])] += 1
        categoryCounts[category(Path(filePath).stem)] += 1
        fileTimes[filePath] += float(case.get("time", 0))

    functions = []
    fileHashes = {}
    exactBodies = defaultdict(list)
    marks = Counter()
    skipSites = []
    importSites = []
    for path in sorted((repoRoot / "tests").glob("*.py")):
        relativePath = str(path.relative_to(repoRoot))
        fileHashes[relativePath] = fingerprint(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                for alias in node.names:
                    target = (module + "." + alias.name) if module else alias.name
                    if target.startswith(("test_", "tests.")):
                        importSites.append({"path": relativePath, "line": node.lineno, "target": target})
            name = callName(node)
            if name.startswith("pytest.mark."):
                marks[name] += 1
            if name in ("pytest.skip", "pytest.importorskip", "pytest.xfail"):
                skipSites.append({"path": relativePath, "line": node.lineno, "call": ast.unparse(node)})
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            bodyKey = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
            calls = [callName(item) for item in ast.walk(node) if isinstance(item, ast.Call)]
            item = {
                "path": relativePath,
                "name": node.name,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "test": node.name.startswith("test"),
                "cases_in_report": caseCounts[(relativePath, node.name)],
                "decorators": [ast.unparse(dec) for dec in node.decorator_list],
                "assert_count": sum(isinstance(item, ast.Assert) for item in ast.walk(node)),
                "raises_count": calls.count("pytest.raises"),
                "numpy_assert_calls": sorted({call for call in calls if call.startswith("np.testing.assert")}),
                "patch_calls": sorted({call for call in calls if "patch" in call}),
            }
            functions.append(item)
            if node.end_lineno - node.lineno >= 3:
                exactBodies[bodyKey].append({key: item[key] for key in ("path", "name", "line", "test")})

    coverage = ET.parse(coveragePath).getroot()
    coverageFiles = []
    for entry in coverage.findall(".//class"):
        lines = entry.findall("./lines/line")
        missed = [int(line.get("number")) for line in lines if int(line.get("hits")) == 0]
        coverageFiles.append({
            "path": entry.get("filename"),
            "line_rate": float(entry.get("line-rate")),
            "executable_lines": len(lines),
            "missed_lines": missed,
        })
    times = [float(case.get("time", 0)) for case in cases]
    result = {
        "scope": "Static inventory plus existing 2026-09-29 Python 3.14 wheel XML; no new execution or mutation testing.",
        "report_hashes": {str(path.relative_to(repoRoot)): fingerprint(path) for path in (junitPath, coveragePath)},
        "summary": {
            "tests": len(cases),
            "files_in_report": len(fileTimes),
            "functions_in_report": len(caseCounts),
            "parameterized_cases": sum("[" in case.get("name") for case in cases),
            "failures": len(junit.findall(".//failure")),
            "errors": len(junit.findall(".//error")),
            "skipped": len(junit.findall(".//skipped")),
            "suite_seconds": sum(float(suite.get("time", 0)) for suite in junit.iter("testsuite")),
            "median_case_seconds": statistics.median(times),
            "cases_under_10ms": sum(value < 0.01 for value in times),
            "coverage": dict(coverage.attrib),
        },
        "categories": dict(categoryCounts),
        "file_seconds": dict(sorted(fileTimes.items(), key=lambda pair: -pair[1])),
        "file_hashes": fileHashes,
        "functions": functions,
        "identical_body_candidates": [group for group in exactBodies.values() if len(group) > 1],
        "called_pytest_marks": dict(marks),
        "skip_sites": skipSites,
        "cross_test_imports": importSites,
        "coverage_files": sorted(coverageFiles, key=lambda item: -len(item["missed_lines"])),
    }
    outputPath = Path(__file__).with_name("0929-test-suite-audit.json")
    outputPath.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("summary", "categories", "identical_body_candidates", "called_pytest_marks", "skip_sites", "cross_test_imports")}, ensure_ascii=False, indent=2))
    print("Most uncovered Python files:")
    for item in result["coverage_files"][:15]:
        print(item["path"], "missed", len(item["missed_lines"]), "line rate", item["line_rate"])


if __name__ == "__main__":
    main()
