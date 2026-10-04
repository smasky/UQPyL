"""Static inventory; counts describe style, not correctness."""
import ast
from collections import Counter
import io
import json
from pathlib import Path
import tokenize

inventory=Counter()
style=Counter()
examples={}
unused=[]
for path in sorted(Path("UQPyL").rglob("*.py")):
    text=path.read_text()
    tree=ast.parse(text,filename=str(path))
    inventory[path.parts[1]]+=1
    lines=text.splitlines()
    trailing=sum(line!=line.rstrip() for line in lines)
    longLines=sum(len(line)>120 for line in lines)
    style["trailing_whitespace_lines"]+=trailing
    style["lines_over_120_chars"]+=longLines
    semicolons=[tok.start[0] for tok in tokenize.generate_tokens(io.StringIO(text).readline)
                if tok.type==tokenize.OP and tok.string==";"]
    style["semicolon_tokens"]+=len(semicolons)
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            style["functions"]+=1
            doc=ast.get_docstring(node) or ""
            if "Args:" in doc:style["google_args_docstrings"]+=1
            if ":param " in doc:style["sphinx_param_docstrings"]+=1
            name=node.name
            if name.startswith("__") and name.endswith("__"):
                if name not in {"__init__","__init_subclass__","__post_init__","__eq__","__ne__","__getitem__",
                                "__setitem__","__len__","__str__","__repr__","__call__","__iter__","__next__",
                                "__enter__","__exit__","__getattr__","__setattr__","__del__","__contains__"}:
                    examples.setdefault("custom_dunder",[]).append(f"{path}:{node.lineno} {name}")
            if node.name=="__init__":
                # Signal only: decorators, locals()/kwargs forwarding and abstract hooks may be legitimate.
                used={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
                arguments=[*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]
                for arg in arguments:
                    if arg.arg not in used and arg.arg!="self":
                        unused.append(f"{path}:{node.lineno} {arg.arg}")
result=dict(files=sum(inventory.values()),module_file_counts=dict(inventory),
            style_counts=dict(style),examples=examples,unused_constructor_argument_signals=unused)
Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
