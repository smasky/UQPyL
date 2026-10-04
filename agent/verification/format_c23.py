"""Apply the pinned formatter and verify executable AST equivalence.

Run from the repository root with conda py312. Only docstrings are excluded
from comparison, since formatting may change their whitespace.
"""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


class WithoutDocs(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body = node.body[1:]
        return node


def fingerprint(text):
    tree = WithoutDocs().visit(ast.parse(text))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def main():
    paths = sorted(Path('UQPyL').rglob('*.py'))
    before = {str(path): path.read_text() for path in paths}
    subprocess.run([sys.executable, '-m', 'ruff', 'format', 'UQPyL'], check=True)
    details = []
    for path in paths:
        old = before[str(path)]
        new = path.read_text()
        if old != new:
            first, second = fingerprint(old), fingerprint(new)
            details.append({'path': str(path), 'before_ast': first, 'after_ast': second})
            assert first == second, f'Executable AST changed: {path}'
    output = {'formatter': 'ruff 0.16.8', 'files_scanned': len(paths),
              'files_formatted': len(details), 'executable_ast_equal': True, 'files': details}
    Path('agent/verification/0919-c23-format.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f'{len(details)} changed files: executable ASTs unchanged.')


if __name__ == '__main__':
    main()
