"""Check regression ASTs, dependencies and collected node IDs after migration."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repoRoot = Path(__file__).resolve().parents[2]
manifestPath = Path(__file__).with_name("0929-test-migration-map.json")


def digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class OriginalImport(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        # The only intentional change inside a moved test function.
        if node.module == "optimization_test_support":
            node.module = "test_optimization_stopping_semantics"
        return node


def main():
    manifest = json.loads(manifestPath.read_text())
    parsed = {}

    def readTree(path):
        if path not in parsed:
            parsed[path] = ast.parse((repoRoot / path).read_text())
        return parsed[path]

    for item in manifest["functions"] + manifest.get("unmoved_functions", []):
        path, name = item["new_id"].split("::")
        matches = [node for node in readTree(path).body if isinstance(node, ast.FunctionDef) and node.name == name]
        assert len(matches) == 1, item
        node = ast.parse(ast.unparse(matches[0])).body[0]
        if name == "testEveryOptimizationConfigurationCanBeRestored":
            node = OriginalImport().visit(node)
        assert digest(node) == item["ast_sha256"], item["new_id"]

    for item in manifest["dependencies"] + manifest["helper_moves"]:
        nodes = readTree(item["new_path"]).body
        assert item["ast_sha256"] in {digest(node) for node in nodes}, item

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=repoRoot,
        text=True,
        capture_output=True,
        check=True,
    )
    nodeIds = [line.strip() for line in collected.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    assert Counter(nodeIds) == Counter(manifest["expected_all_nodeids"]), (
        "Collected cases differ from the exact migration map"
    )
    assert len(nodeIds) == manifest["baseline_cases"] == 2005
    for tree in parsed.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("test_"), ast.unparse(node)
    print(
        f"Verified {len(manifest['functions'])} migrated function ASTs, {len(manifest['cases'])} case mappings, dependencies and all {len(nodeIds)} collected IDs."
    )
    print("Only intentional in-function change: configuration restore imports the shared factory module.")


if __name__ == "__main__":
    main()
