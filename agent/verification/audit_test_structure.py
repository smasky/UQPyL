"""Inventory test definitions and exact AST duplicate bodies; no semantic deletion advice."""
import ast
from collections import defaultdict
import json
from pathlib import Path


def inventory():
    groups = defaultdict(list)
    files = []
    for path in sorted(Path('tests').glob('*.py')):
        tree = ast.parse(path.read_text())
        tests = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            key = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
            if len(body) > 1:
                groups[key].append({'path': str(path), 'name': node.name, 'line': node.lineno})
            if node.name.startswith('test'):
                tests.append(node.name)
        files.append({'path': str(path), 'test_definitions': tests})
    return {'files': files, 'test_definition_count': sum(len(f['test_definitions']) for f in files),
            'duplicate_body_candidates': [v for v in groups.values() if len(v) > 1]}


if __name__ == '__main__':
    print(json.dumps(inventory(), indent=2))
