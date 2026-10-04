"""Apply the configured formatter and verify executable AST preservation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def executableTree(path):
    tree=ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body=node.body[1:]
    return ast.dump(tree,include_attributes=False)

paths=list(Path('UQPyL').rglob('*.py'))
before={str(p):(hashlib.sha256(p.read_bytes()).hexdigest(),executableTree(p)) for p in paths}
subprocess.run(['/home/wmtsky/anaconda3/envs/py312/bin/python','-m','ruff','format','UQPyL'],check=True)
changed=[]
for p in paths:
    digest,tree=before[str(p)]
    assert executableTree(p)==tree,str(p)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        changed.append(str(p))
Path('agent/verification/1004-format-ast.json').write_text(json.dumps(dict(changed_files=changed,executable_ast_unchanged=True),indent=2))
print('AST preserved;',len(changed),'files formatted')
