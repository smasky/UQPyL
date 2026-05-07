from pathlib import Path

import pytest


def test_problem_benchmark_modules_import_without_syntaxwarning():
    files = [
        "UQPyL/problem/sop/single_simple_problem.py",
        "UQPyL/problem/sop/single_constraint_problem.py",
    ]

    root = Path(__file__).resolve().parents[1]
    for relative_path in files:
        source_path = root / relative_path
        source = source_path.read_text(encoding="utf-8")
        with pytest.warns(None) as record:
            compile(source, str(source_path), "exec")
        syntax_warnings = [warning for warning in record if warning.category is SyntaxWarning]
        assert syntax_warnings == []
