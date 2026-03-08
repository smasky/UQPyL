import ast
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

def _check_file_exists(file: str):
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")

# -------------------------
# 1. basic
# -------------------------
@dataclass
class BasicSpec:
    projectPath: str
    workPath: str
    basePath: str
    command: str
    timeout: int = -1
    parallel: int = 1

    @staticmethod
    def from_dict(d: Dict[str, Any], base_path: Path) -> "BasicSpec":
        if not isinstance(d, dict):
            raise ValueError("basic must be a mapping/object")
        
        for k in ("projectPath", "workPath", "command"):
            if not d.get(k):
                raise ValueError(f"basic.{k} is required")
            
        return BasicSpec(
            basePath=str(base_path).replace("\\", "/"),
            projectPath=str(d["projectPath"].replace("\\", "/")),
            workPath=str(d["workPath"].replace("\\", "/")),
            command=str(d["command"]),
            timeout=int(d.get("timeout", -1)),
            parallel=int(d.get("parallel", 1)),
        )

# -------------------------
# parameters
# -------------------------

@dataclass
class ParametersSpec:
    library: str
    params: str
    hardBound: bool = True
    func: Optional[str] = None
    physical: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any], base_path: str) -> "ParametersSpec":
        
        if not isinstance(d, dict):
            raise ValueError("parameters must be a mapping/object")
        
        library = (base_path / d.get("library")).as_posix()
        params = (base_path / d.get("params")).as_posix()
        
        _check_file_exists(library)
        _check_file_exists(params)
        
        if not library or not params:
            raise ValueError("parameters.library and parameters.params are required")

        func = d.get("func")
        physical = d.get("physical")
        if func and not physical:
            raise ValueError("parameters.func requires parameters.physical")
        if physical and not func:
            raise ValueError("parameters.physical requires parameters.func")

        if physical:
            physical = (base_path / physical).as_posix()
            _check_file_exists(physical)
        
        return ParametersSpec(library=str(library), params=str(params), func=func, physical=physical, hardBound=bool(d.get("hardBound", True)))

    def get_env_dep_list(self):
        
        with open(self.params, "r") as f:
            raw_data = yaml.safe_load(f)
            
        param_list = raw_data.get("design_parameters", [])
        
        env_list = []
        for item in param_list:
            if "cache" in item and item["cache"]:
                env_list.append(item["name"])
        
        return env_list, []
    
@dataclass
class CallSpec:
    func: str
    args: Dict[str, Any]

# -------------------------
# 2. Series extraction
# -------------------------
def _expr_vars(expr: str):
    """Extract variable names used in expr (tn1, tn2, ...)."""
    tree = ast.parse(expr, mode="eval")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

def _expand_row_ranges(row_ranges: List[List[int]]) -> List[int]:
    out = []
    for rr in row_ranges:
        if len(rr) == 2:
            a, b = rr
            step = 1
        else:
            a, b, step = rr
        out.extend(range(int(a), int(b) + 1, int(step)))
    out.sort()
    return out

ColumnKind = Literal["span", "list"]

@dataclass
class ColumnSpec:
    kind: ColumnKind
    span: Optional[Tuple[int, int]] = None
    col: Optional[int] = None
    delimiter: str = "whitespace"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ColumnSpec":
        has_span = "colSpan" in d
        has_num = "colNum" in d
        
        if has_span == has_num:
            raise ValueError("Must set exactly one of colSpan or colNum.")

        if has_span:
            cs = d["colSpan"]
            if not (isinstance(cs, list) and len(cs) == 2):
                raise ValueError(f"colSpan must be [start,end], got: {cs}")
            a, b = int(cs[0]), int(cs[1])
            return ColumnSpec(kind="span", span=(a, b))

        cl = d["colNum"]
        if not isinstance(cl, int):
            raise ValueError(f"colNum must be an int, got: {cl}")

        return ColumnSpec(kind="list", col=cl, delimiter=d.get("delimiter", "whitespace"))
    
@dataclass
class ExprSpec:
    expr: str
    size: int
    deps: List[str]
    
    @staticmethod
    def from_dict(d: Dict[str, Any]):
        expr = d.get("expr")
        if not expr or not isinstance(expr, str):
            raise ValueError(f"sim.expr must be a non-empty string, got: {expr}")       
        return ExprSpec(expr=expr, size=-1, deps= _expr_vars(expr))

@dataclass
class ExtractSpec:
    file: str
    rows: List[int]
    column: ColumnSpec
    size: int

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExtractSpec":
        if not isinstance(d, dict):
            raise ValueError(f"ExtractSpec must be a mapping/object, got: {type(d)}")

        if "file" not in d:
            raise ValueError(f"ExtractSpec missing 'file': {d}")

        rr = d.get("rowRanges", [])
        if rr and not isinstance(rr, list):
            raise ValueError(f"rowRanges must be a list, got: {rr}")

        row_ranges: List[List[int]] = []
        for item in rr:
            if not isinstance(item, list) or len(item) not in (2, 3):
                raise ValueError(f"rowRanges item must be [start,end] or [start,end,step], got: {item}")
            row_ranges.append([int(x) for x in item])

        expanded_ranges = _expand_row_ranges(row_ranges) if row_ranges else []
        
        rl = d.get("rowList", [])
        if rl and (not isinstance(rl, list) or not all(isinstance(x, int) for x in rl)):
            raise ValueError(f"rowList must be a list of integers, got: {rl}")
            
        row_list = [int(x) for x in rl]
        
        merged_rows = sorted(list(set(expanded_ranges + row_list)))

        if not merged_rows:
            raise ValueError(f"ExtractSpec in '{d['file']}' must define at least one row via 'rowRanges' or 'rowList'")
        
        size = len(merged_rows)
        ss = int(d.get("size", size))
        
        if ss != size:
            raise ValueError(f"ExtractSpec size mismatch: you set size={ss} but the actual size is {size}")
        
        col = ColumnSpec.from_dict(d)
        
        return ExtractSpec(file=str(d["file"]), rows=merged_rows, size=size, column=col)

@dataclass
class SeriesSpec:
    id: str
    name: str
    sim: Union[ExprSpec, ExtractSpec, CallSpec]
    obs: Optional[ExtractSpec]
    cache: bool = False
    size: int = -1
    @staticmethod
    def from_dict(d: Dict[str, Any], base_path: Path) -> "SeriesSpec":
        if not isinstance(d, dict):
            raise ValueError("SeriesSpec must be a mapping/object")

        sid = d.get("id")
        if not sid:
            raise ValueError(f"SeriesSpec missing id: {d}")
        
        sim_raw = d.get("sim")
        if not isinstance(sim_raw, dict):
             raise ValueError(f"Series {sid}: 'sim' block must be a dictionary.")
        
        if "call" in sim_raw:
            c_data = sim_raw["call"]
            sim = CallSpec(func=c_data["func"], args=c_data.get("args", {}))
        elif "expr" in sim_raw:
            sim = ExprSpec.from_dict(sim_raw)
        else:
            sim = ExtractSpec.from_dict(sim_raw)
        
        obs_raw = d.get("obs")
        obs = None if obs_raw is None else ExtractSpec.from_dict(obs_raw)

        if obs:
            obs.file = (base_path / obs.file).as_posix()
            _check_file_exists(obs.file)
            
        return SeriesSpec(id=str(sid), name=str(d.get("desc", sid)), sim=sim, obs=obs, cache=d.get("cache", False))

    def get_env_dep_list(self):
        env = []; dep = []
        
        # env
        if self.cache:
            env.append(f"{self.id}_sim")
        if self.obs:
            env.append(f"{self.id}_obs")
            
        # dep
        if isinstance(self.sim, ExprSpec):
            dep += self.sim.deps
        elif isinstance(self.sim, CallSpec):
            dep += list(self.sim.args.values())
                    
        return env, dep
    
# ==========================================
# 3. Functions & Derived
# ==========================================
@dataclass
class FunctionSpec:
    name: str
    kind: Literal["builtin", "external"]
    args: List[str]
    file: Optional[str] = None
    
    @staticmethod
    def from_dict(d: Dict[str, Any], base_path: Path) -> "FunctionSpec":
        name = d.get("name")
        if not name:
            raise ValueError(f"Function missing name: {d}")
        
        kind = d.get("kind")
        if kind not in ("builtin", "external"):
            raise ValueError(f"Function kind must be builtin or external, got: {kind}")
        
        args = d.get("args", [])
        if not isinstance(args, list):
            raise ValueError(f"Function args must be a list, got: {args}")
            
        file = d.get("file")
        if file:
            file = (base_path / file).as_posix()
            _check_file_exists(file)
        
        return FunctionSpec(name=name, kind=kind, args=args, file=file)

@dataclass
class DerivedSpec:
    id: str
    desc: str
    call: Optional[CallSpec] = None
    expr: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DerivedSpec":
        did = d.get("id")
        if not did:
            raise ValueError("Derived item missing id")

        call_raw = d.get("call")
        expr_raw = d.get("expr")

        if call_raw and expr_raw:
            raise ValueError(f"Derived {did} cannot have both 'call' and 'expr'")
        if not call_raw and not expr_raw:
            raise ValueError(f"Derived {did} must have either 'call' or 'expr'")

        call_spec = None
        if call_raw:
            if not isinstance(call_raw, dict) or "func" not in call_raw or "args" not in call_raw:
                raise ValueError(f"Derived {did} call must contain 'func' and 'args'")
            call_spec = CallSpec(func=call_raw["func"], args=call_raw["args"])

        return DerivedSpec(
            id=str(did),
            desc=str(d.get("desc", did)),
            call=call_spec,
            expr=str(expr_raw) if expr_raw else None
        )

    def get_env_dep_list(self):
        env = []; dep = []
        
        # env
        env.append(self.id)
        
        # dep
        if self.call:
            dep += list(self.call.args.values())
        # TODO
        elif self.expr:
            dep += self.expr.deps
        return env, dep
    
# ==========================================
# 4. Objectives & Diagnostics
# ==========================================

@dataclass
class ObjectiveSpec:
    id: str
    desc: str
    sense: Literal["max", "min"]
    ref: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObjectiveSpec":
        oid = d.get("id")
        if not oid:
            raise ValueError("Objective missing id")
            
        ref = d.get("ref")
        if not ref:
            raise ValueError(f"Objective {oid} missing 'ref'")
            
        return ObjectiveSpec(
            id=str(oid), 
            desc=str(d.get("desc", oid)), 
            sense=d.get("sense", "min"), 
            ref=str(ref)
        )
    
    def get_env_dep_list(self):
        env = []; dep = []
        
        # env
        env.append(self.id)
        
        # dep
        if self.ref not in env:
            dep.append(self.ref)
        return env, dep

@dataclass
class ConstraintSpec:
    id: str
    desc: str
    ref: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConstraintSpec":
        oid = d.get("id")
        if not oid:
            raise ValueError("Constraint missing id")
            
        ref = d.get("ref")
        if not ref:
            raise ValueError(f"Constraint {oid} missing 'ref'")
            
        return ConstraintSpec(
            id=str(oid), 
            desc=str(d.get("desc", oid)), 
            ref=str(ref)
        )
    
    def get_env_dep_list(self):
        env = []; dep = []
        
        # env
        env.append(self.id)
        
        # dep
        if self.ref not in env:
            dep.append(self.ref)
        return env, dep
    
    
@dataclass
class DiagnosticSpec:
    id: str
    name: str
    ref: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DiagnosticSpec":
        did = d.get("id")
        ref = d.get("ref")
        if not did or not ref:
            raise ValueError("Diagnostic must have 'id' and 'ref'")
        return DiagnosticSpec(id=str(did), name=str(d.get("name", did)), ref=str(ref))
    
    def get_env_dep_list(self):
        env = []; dep = []
        
        # env
        env.append(self.id)
        
        # dep
        if self.ref not in env:
            dep.append(self.ref)
        return env, dep

@dataclass(frozen=True)
class ObjectiveBlock:
    use: List[str]
    items: Dict[str, ObjectiveSpec]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObjectiveBlock":
        items_raw = d.get("items", [])
        items = {o.id: o for o in [ObjectiveSpec.from_dict(x) for x in items_raw]}

        ids = list(items.keys())
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate objective id in objectives.items")

        use = d.get("use", ids)
        for oid in use:
            if oid not in items:
                raise ValueError(f"objectives.use references unknown objective id: {oid}")

        return ObjectiveBlock(use=use, items=items)
@dataclass
class ConstraintBlock:
    use: List[str]
    items: Dict[str, ConstraintSpec]
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConstraintBlock":
        items_raw = d.get("items", [])
        items = {c.id: c for c in [ConstraintSpec.from_dict(x) for x in items_raw]}
        
        ids = list(items.keys())
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate constraint id in constraints.items")
        
        use = d.get("use", ids)
        for cid in use:
            if cid not in items:
                raise ValueError(f"constraints.use references unknown constraint id: {cid}")
                
        return ConstraintBlock(use=use, items=items)

@dataclass(frozen=True)
class DiagnosticBlock:
    use: List[str]
    items: Dict[str, DiagnosticSpec]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DiagnosticBlock":
        items_raw = d.get("items", [])
        items = {diag.id: diag for diag in [DiagnosticSpec.from_dict(x) for x in items_raw]}

        ids = list(items.keys())
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate diagnostic id in diagnostics.items")

        use = d.get("use", ids)
        for did in use:
            if did not in items:
                raise ValueError(f"diagnostics.use references unknown diagnostic id: {did}")

        return DiagnosticBlock(use=use, items=items)

# ==========================================
# 5. Reporter
# ==========================================
@dataclass
class ReporterSpec:
    flush_interval: int
    parameters: Union[str, List[str]]
    scalars: List[str]
    series: List[str]
    output_series_csv: bool

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        if not isinstance(d, dict):
            d = {}
            
        return ReporterSpec(
            flush_interval=int(d.get("flush_interval", 50)),
            parameters=d.get("parameters", "all"),
            scalars=d.get("scalars", []),
            series=d.get("series", []),
            output_series_csv=bool(d.get("output_series_csv", False))
        )
# -------------------------
# RunConfig
# -------------------------

@dataclass
class RunConfig:
    version: str
    basic: BasicSpec
    parameters: ParametersSpec
    series: List[SeriesSpec]
    functions: Dict[str, FunctionSpec]
    derived: List[DerivedSpec]
    objectives: ObjectiveBlock
    constraints: ConstraintBlock
    diagnostics: DiagnosticBlock
    reporter: ReporterSpec
    series_index: Dict[str, SeriesSpec]
        
    @staticmethod
    def from_dict(yaml_file: Path):
        
        cfg = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        
        base_path = yaml_file.parent
    
        if not isinstance(cfg, dict):
            raise ValueError("YAML root must be a mapping/object")
        
        version = cfg.get("version")
        if version != "general":
            raise ValueError(f"Unsupported version: {version}")
        
        # basic setting
        basic = BasicSpec.from_dict(cfg.get("basic", {}), base_path)
        # parameters setting
        parameters = ParametersSpec.from_dict(cfg.get("parameters", {}), base_path)
        # reporter setting
        reporter = ReporterSpec.from_dict(cfg.get("reporter", {}))
        # series setting
        series_list = [SeriesSpec.from_dict(s, base_path) for s in cfg.get("series", [])]
        if not series_list:
            raise ValueError("series must be a non-empty list")

        series_index: Dict[str, SeriesSpec] = {}
        for s in series_list:
            if s.id in series_index:
                raise ValueError(f"Duplicate series id: {s.id}")
            series_index[s.id] = s
                    
        funcs_raw = cfg.get("functions", [])
        if not isinstance(funcs_raw, list):
            raise ValueError("functions must be a list")
            
        functions = {}
        for f_dict in funcs_raw:
            f_spec = FunctionSpec.from_dict(f_dict, base_path)
            if f_spec.name in functions:
                raise ValueError(f"Duplicate function name: {f_spec.name}")
            functions[f_spec.name] = f_spec

        derived = [DerivedSpec.from_dict(d) for d in cfg.get("derived", [])]
        objectives = ObjectiveBlock.from_dict(cfg.get("objectives", {}))
        constraints = ConstraintBlock.from_dict(cfg.get("constraints", {}))
        diagnostics = DiagnosticBlock.from_dict(cfg.get("diagnostics", {}))
 
        # Fail-Fast
        env = ["X", "i"] ; dep = []
        e, d = parameters.get_env_dep_list()
        env += e
        dep += d
        
        for s in series_list:
            e, d = s.get_env_dep_list()
            env += e
            dep += d
        
        for d in derived:
            e, d = d.get_env_dep_list()
            env += e
            dep += d
        
        for o in objectives.items.values():
            e, d = o.get_env_dep_list()
            env += e
            dep += d
        
        for c in constraints.items.values():
            e, d = c.get_env_dep_list()
            env += e
            dep += d
        
        for d in diagnostics.items.values():
            e, d = d.get_env_dep_list()
            env += e
            dep += d
        
        env = list(set(env))
        dep = list(set(dep))
        
        for d in dep:
            if d not in env:
                raise ValueError(f"Dependency '{d}' is not in environment")
        
        return RunConfig(
            version=str(version),
            basic=basic,
            parameters=parameters,
            series=series_list,
            functions=functions,
            derived=derived,
            objectives=objectives,
            constraints=constraints,
            diagnostics=diagnostics,
            reporter=reporter,
            series_index=series_index
        )

def load_config(path: str) -> RunConfig:
    
    yaml_file = Path(path).resolve()

    cfg = RunConfig.from_dict(yaml_file)
    
    return cfg