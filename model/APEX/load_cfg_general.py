from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal, get_args

import os

import yaml

ColumnKind = Literal["span", "list"]
MetricKind = Literal["R2", "MSE", "NSE", "KGE", "MAE"]
_METRICS = set(get_args(MetricKind))

# -------------------------
# basic / parameters
# -------------------------

@dataclass
class BasicSpec:
    projectPath: str
    workPath: str
    exeName: str
    parallel: int = 1
    hardBound: bool = True

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BasicSpec":
        if not isinstance(d, dict):
            raise ValueError("basic must be a mapping/object")

        for k in ("projectPath", "workPath", "exeName"):
            if not d.get(k):
                raise ValueError(f"basic.{k} is required")

        
        
        return BasicSpec(
            projectPath=str(d["projectPath"].replace("\\", "/")),
            workPath=str(d["workPath"].replace("\\", "/")),
            exeName=str(d["exeName"]),
            parallel=int(d.get("parallel", 1)),
            hardBound=bool(d.get("hardBound", True)),
        )

@dataclass
class ParametersSpec:
    info: str
    param: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ParametersSpec":
        if not isinstance(d, dict):
            raise ValueError("parameters must be a mapping/object")

        info = d.get("info")
        param = d.get("param")
        if not info or not param:
            raise ValueError("parameters.info and parameters.param are required")

        return ParametersSpec(info=str(info), param=str(param))

# -------------------------
# series extraction
# -------------------------

@dataclass
class ColumnSpec:
    kind: ColumnKind
    span: Optional[Tuple[int, int]] = None
    cols: Optional[List[int]] = None
    delimiter: str = "whitespace"  # for kind="list"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ColumnSpec":
        has_span = "colSpan" in d
        has_list = "colList" in d
        if has_span == has_list:
            raise ValueError("Must set exactly one of colSpan or colList.")

        if has_span:
            cs = d["colSpan"]
            if not (isinstance(cs, list) and len(cs) == 2):
                raise ValueError(f"colSpan must be [start,end], got: {cs}")
            a, b = int(cs[0]), int(cs[1])
            return ColumnSpec(kind="span", span=(a, b))

        cl = d["colList"]
        if not (isinstance(cl, list) and all(isinstance(x, int) for x in cl)):
            raise ValueError(f"colList must be a list of int, got: {cl}")

        delim = d.get("delimiter", "whitespace")
        return ColumnSpec(kind="list", cols=[int(x) for x in cl], delimiter=str(delim))

@dataclass
class ExtractSpec:
    file: str
    rowRanges: List[List[int]]
    column: ColumnSpec

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExtractSpec":
        if not isinstance(d, dict):
            raise ValueError(f"ExtractSpec must be a mapping/object, got: {type(d)}")

        if "file" not in d:
            raise ValueError(f"ExtractSpec missing 'file': {d}")

        rr = d.get("rowRanges")
        if not isinstance(rr, list) or not rr:
            raise ValueError(f"rowRanges must be a non-empty list, got: {rr}")

        row_ranges: List[List[int]] = []
        for item in rr:
            if not isinstance(item, list) or len(item) not in (2, 3):
                raise ValueError(f"rowRanges item must be [start,end] or [start,end,step], got: {item}")
            row_ranges.append([int(x) for x in item])

        col = ColumnSpec.from_dict(d)
        return ExtractSpec(file=str(d["file"]), rowRanges=row_ranges, column=col)

@dataclass
class SeriesSpec:
    id: str
    name: str
    sim: ExtractSpec
    obs: Optional[ExtractSpec]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SeriesSpec":
        if not isinstance(d, dict):
            raise ValueError("SeriesSpec must be a mapping/object")

        sid = d.get("id")
        if not sid:
            raise ValueError(f"SeriesSpec missing id: {d}")

        sim = ExtractSpec.from_dict(d.get("sim", {}))
        obs_raw = d.get("obs")
        obs = None if obs_raw is None else ExtractSpec.from_dict(obs_raw)

        return SeriesSpec(id=str(sid), name=str(d.get("name", sid)), sim=sim, obs=obs)

# -------------------------
# objectives / diagnostics
# -------------------------

@dataclass
class Term:
    id: str          # series id (s1/s2...)
    weight: float = 1.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Term":
        if not isinstance(d, dict):
            raise ValueError(f"Term must be a mapping/object, got: {type(d)}")
        sid = d.get("id")
        if not sid:
            raise ValueError(f"Term missing id: {d}")
        return Term(id=str(sid), weight=float(d.get("weight", 1.0)))

@dataclass
class ObjectiveSpec:
    id: str
    name: str
    metric: MetricKind
    direction: Literal["maximize", "minimize"]
    reduce: str
    terms: List[Term] 

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObjectiveSpec":
        if not isinstance(d, dict):
            raise ValueError("ObjectiveSpec must be a mapping/object")

        oid = d.get("id")
        if not oid:
            raise ValueError(f"Objective missing id: {d}")

        metric = d.get("metric")
        if not metric:
            raise ValueError(f"Objective {oid} missing metric")
        if metric not in _METRICS:
            raise ValueError(f"Objective {oid} metric must be one of {_METRICS}, got: {metric}")

        direction = str(d.get("direction", "min")).lower()
        if direction not in ("max", "min"):
            raise ValueError(f"Objective {oid} direction must be max|min, got: {direction}")

        reduce = str(d.get("reduce", "weighted_mean"))

        raw_terms = d.get("series") or []
        if not isinstance(raw_terms, list):
            raise ValueError(f"Objective {oid} series must be a list, got: {type(raw_terms)}")
        terms = [Term.from_dict(t) for t in raw_terms]  # 允许 []

        return ObjectiveSpec(
            id=str(oid),
            name=str(d.get("name", oid)),
            metric=metric,          # type: ignore
            direction=direction,    # type: ignore
            reduce=reduce,
            terms=terms,
        )

@dataclass(frozen=True)
class DiagnosticSpec:
    id: str
    name: str
    metric: MetricKind
    reduce: str
    terms: List[Term]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DiagnosticSpec":
        if not isinstance(d, dict):
            raise ValueError("DiagnosticSpec must be a mapping/object")

        did = d.get("id")
        if not did:
            raise ValueError(f"Diagnostic missing id: {d}")

        metric = d.get("metric")
        if not metric:
            raise ValueError(f"Diagnostic {did} missing metric")
        if metric not in _METRICS:
            raise ValueError(f"Diagnostic {did} metric must be one of {_METRICS}, got: {metric}")

        raw_terms = d.get("series") or []
        if not isinstance(raw_terms, list):
            raise ValueError(f"Diagnostic {did} series must be a list, got: {type(raw_terms)}")
        terms = [Term.from_dict(t) for t in raw_terms]  # 允许 []

        reduce = str(d.get("reduce", "weighted_mean"))
        
        return DiagnosticSpec(
            id=str(did),
            name=str(d.get("name", did)),
            metric=metric,  # type: ignore
            reduce=reduce,
            terms=terms,
        )

@dataclass(frozen=True)
class ObjectiveBlock:
    use: List[str]
    items: Dict[str, ObjectiveSpec]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ObjectiveBlock":
        if not isinstance(d, dict):
            raise ValueError("objectives must be a mapping with keys: use, items")

        items_raw = d.get("items")
        if not isinstance(items_raw, list):
            raise ValueError("objectives.items must be a list")
        
        items = {o.id: o for o in [ObjectiveSpec.from_dict(x) for x in items_raw]}

        # unique id
        ids = list(items.keys())
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate objective id in objectives.items")

        # use defaults to all items
        use = d.get("use", ids)
        if not isinstance(use, list) or not all(isinstance(x, str) for x in use):
            raise ValueError("objectives.use must be a list of strings")

        for oid in use:
            if oid not in set(ids):
                raise ValueError(f"objectives.use references unknown objective id: {oid}")

        return ObjectiveBlock(use=use, items=items)

@dataclass(frozen=True)
class DiagnosticBlock:
    use: List[str]
    items: List[DiagnosticSpec]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DiagnosticBlock":
        if not isinstance(d, dict):
            raise ValueError("diagnostics must be a mapping with keys: use, items")

        items_raw = d.get("items")
        if not isinstance(items_raw, list):
            raise ValueError("diagnostics.items must be a list")
        items = {d.id: d for d in [DiagnosticSpec.from_dict(x) for x in items_raw]}

        ids = list(items.keys())
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate diagnostic id in diagnostics.items")

        use = d.get("use", ids)
        if not isinstance(use, list) or not all(isinstance(x, str) for x in use):
            raise ValueError("diagnostics.use must be a list of strings")

        for did in use:
            if did not in set(ids):
                raise ValueError(f"diagnostics.use references unknown diagnostic id: {did}")

        return DiagnosticBlock(use=use, items=items)

# -------------------------
# RunConfig
# -------------------------

@dataclass(frozen=True)
class RunConfig:
    version: str
    basic: BasicSpec
    parameters: ParametersSpec
    series: List[SeriesSpec]
    objectives: ObjectiveBlock
    diagnostics: DiagnosticBlock

    series_index: Dict[str, SeriesSpec]
    objective_index: Dict[str, ObjectiveSpec]
    diagnostic_index: Dict[str, DiagnosticSpec]

    def update_validate_files(self):
        
        workPath = self.basic.workPath
        self.check_dir_exists(workPath)
        
        self.parameters.info = os.path.join(workPath, self.parameters.info)
        self.parameters.param = os.path.join(workPath, self.parameters.param)
        self.check_file_exists(self.parameters.info)
        self.check_file_exists(self.parameters.param)
        
        for series in self.series:
            if series.obs:
                self.check_file_exists(os.path.join(workPath, series.obs.file))
        
    def check_file_exists(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
    
    def check_dir_exists(self, dir: str):
        if not os.path.exists(dir):
            raise FileNotFoundError(f"Directory not found: {dir}")
        
    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "RunConfig":
        version = cfg.get("version")
        if version != "general":
            raise ValueError(f"Unsupported version: {version}")

        basic = BasicSpec.from_dict(cfg.get("basic", {}))
        parameters = ParametersSpec.from_dict(cfg.get("parameters", {}))

        raw_series = cfg.get("series")
        if not isinstance(raw_series, list) or not raw_series:
            raise ValueError("series must be a non-empty list")
        series_list = [SeriesSpec.from_dict(s) for s in raw_series]

        series_index: Dict[str, SeriesSpec] = {}
        for s in series_list:
            if s.id in series_index:
                raise ValueError(f"Duplicate series id: {s.id}")
            series_index[s.id] = s

        objectives = ObjectiveBlock.from_dict(cfg.get("objectives", {}))
        diagnostics = DiagnosticBlock.from_dict(cfg.get("diagnostics", {}))

        objective_index = objectives.items
        diagnostic_index = diagnostics.items

        for oid in objectives.use:
            o = objective_index[oid]
            for t in o.terms:
                if t.id not in series_index:
                    raise ValueError(f"Objective {o.id} references unknown series: {t.id}")

        for did in diagnostics.use:
            d = diagnostic_index[did]
            for t in d.terms:
                if t.id not in series_index:
                    raise ValueError(f"Diagnostic {d.id} references unknown series: {t.id}")

        return RunConfig(
            version=str(version),
            basic=basic,
            parameters=parameters,
            series=series_list,
            objectives=objectives,
            diagnostics=diagnostics,
            series_index=series_index,
            objective_index=objective_index,
            diagnostic_index=diagnostic_index,
        )

def load_config(path: str):
    
    raw_cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    
    if not isinstance(raw_cfg, dict):
        raise ValueError("YAML root must be a mapping/object")
    
    cfg = RunConfig.from_dict(raw_cfg)
    
    cfg.update_validate_files()
    
    return cfg