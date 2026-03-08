import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional, Tuple
from datetime import datetime

import yaml
import numpy as np

from write_in_handler import WriteInHandler
from run_error import RunError

# -------------------------------------------------------------
# Constant Mappings
# -------------------------------------------------------------
TYPE_MAP = {"float": 0, "int": 1, "discrete": 2}
MODE_MAP = {"r": 0, "v": 1, "a": 2}  # r: relative, v: value (replace), a: add

# -------------------------------------------------------------
# Data Classes
# -------------------------------------------------------------
def _expand_row_ranges(row_ranges: List[List[int]]) -> List[int]:
    out: List[int] = []
    for rr in row_ranges:
        if len(rr) == 2:
            a, b = rr
            step = 1
        elif len(rr) == 3:
            a, b, step = rr
        else:
            raise ValueError(f"Invalid row range: {rr}")

        out.extend(range(int(a), int(b) + 1, int(step)))

    out.sort()
    return out


@dataclass
class ParamFileSpec:
    """Location info within a file."""
    name: str
    line: Union[int, List[int]]
    start: int
    width: int
    precision: int
    maxNum: int


@dataclass
class ParamSpec:
    """Represents a static physical parameter definition in the Library."""
    name: str
    type: int
    bounds: List[float]
    file: ParamFileSpec

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ParamSpec":
        if "name" not in d:
            raise ValueError("Library parameter missing 'name'")
        if "type" not in d:
            raise ValueError(f"Library parameter '{d.get('name', '<unknown>')}' missing 'type'")
        if "bounds" not in d:
            raise ValueError(f"Library parameter '{d.get('name', '<unknown>')}' missing 'bounds'")
        if "file" not in d:
            raise ValueError(f"Library parameter '{d.get('name', '<unknown>')}' missing 'file'")

        raw_line = d["file"]["line"]
        if isinstance(raw_line, list) and len(raw_line) > 0 and isinstance(raw_line[0], list):
            parsed_line = _expand_row_ranges(raw_line)
        else:
            parsed_line = raw_line

        return ParamSpec(
            name=d["name"],
            type=TYPE_MAP.get(d["type"], 0),
            bounds=d["bounds"],
            file=ParamFileSpec(
                name=d["file"]["name"],
                line=parsed_line,
                start=int(d["file"]["start"]),
                width=int(d["file"]["width"]),
                precision=int(d["file"]["precision"]),
                maxNum=int(d["file"].get("maxNum", 1)),
            ),
        )

    @property
    def lb(self) -> float:
        return float(self.bounds[0])

    @property
    def ub(self) -> float:
        return float(self.bounds[1])


@dataclass
class DesignParamSpec:
    """Represents a Design Variable (X) in the Optimization Space."""
    name: str
    index: int
    type: int
    bounds: List[float]
    sets: List[float]
    cache: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any], index: int) -> "DesignParamSpec":
        p_name = d.get("name")
        if not p_name:
            raise ValueError(f"Design parameter at index {index} missing 'name'")

        p_type = TYPE_MAP.get(d.get("type"), 0)

        if p_type == 2 and "sets" not in d:
            raise ValueError(f"Discrete variable '{p_name}' missing 'sets'")
        if p_type in (0, 1) and "bounds" not in d:
            raise ValueError(f"Continuous variable '{p_name}' missing 'bounds'")

        bounds = [0, 1] if p_type == 2 else d.get("bounds", [0, 1])

        return DesignParamSpec(
            name=p_name,
            index=index,
            type=p_type,
            bounds=bounds,
            sets=d.get("sets", []),
            cache=bool(d.get("cache", False)),
        )

    @property
    def lb(self) -> float:
        return float(self.bounds[0])

    @property
    def ub(self) -> float:
        return float(self.bounds[1])


@dataclass
class PhysicalParamSpec:
    """Represents a Physical Parameter (P) that will be written to files."""
    index: int      # Index in the source array (P array or X array)
    name: str       # Name in the Library
    mode: int       # r/v/a
    cache: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any], index: int) -> "PhysicalParamSpec":
        if "name" not in d:
            raise ValueError(f"Physical parameter at index {index} missing 'name'")

        return PhysicalParamSpec(
            index=index,
            name=d["name"],
            mode=MODE_MAP.get(d.get("mode", "v"), 1),  # default to 'v'
            cache=bool(d.get("cache", False)),
        )


# -------------------------------------------------------------
# Parameter Manager
# -------------------------------------------------------------
class ParamManager:
    def __init__(self, cfg, functionManager, backupPath):
        self.cfg = cfg
        self.func_manager = functionManager
        self.backup_path = Path(backupPath) if backupPath else None

        self.writeInTask: Dict[str, Dict[str, Any]] = {}
        self.cached_indices: Dict[str, int] = {}      # Cache for X
        self.cached_indices_phy: Dict[str, int] = {}  # Cache for P

        # 1. Read Library
        self.library: Dict[str, ParamSpec] = self._read_library()

        # 2. Check for Transform Function
        self.transform_func: Optional[str] = None
        if getattr(self.cfg.parameters, "func", None):
            self.transform_func = self.cfg.parameters.func

        # 3. Unified Initialization
        self.nInput, self.xLabels, self.varType, self.varSet, self.ub, self.lb = self._init_params()

    # -------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    def _read_library(self) -> Dict[str, ParamSpec]:
        lib_path = Path(self.cfg.parameters.library)
        raw_data = self._load_yaml(lib_path)

        library_dict: Dict[str, ParamSpec] = {}
        for item in raw_data.get("parameter_library", []):
            spec = ParamSpec.from_dict(item)
            if spec.name in library_dict:
                raise ValueError(f"Duplicate parameter name in library: {spec.name}")
            library_dict[spec.name] = spec

        return library_dict

    def _init_params(self) -> Tuple[int, List[str], List[int], Dict[int, List[float]], List[float], List[float]]:
        """
        Unified logic to initialize Design Variables (X) and Physical Parameters (P).
        """

        # --- Step A: Process Design Variables (X) ---
        design_path = Path(self.cfg.parameters.params)
        design_raw_data = self._load_yaml(design_path)
        design_list_raw = design_raw_data.get("design_parameters", [])

        names: List[str] = []
        seen_names = set()

        types: List[int] = []
        ubs: List[float] = []
        lbs: List[float] = []
        sets: Dict[int, List[float]] = {}

        for i, item in enumerate(design_list_raw):
            spec = DesignParamSpec.from_dict(item, i)

            if spec.name in seen_names:
                raise ValueError(f"Duplicate design parameter name: {spec.name}")
            seen_names.add(spec.name)

            names.append(spec.name)
            types.append(spec.type)
            if spec.type == 2:
                sets[i] = spec.sets
            lbs.append(spec.lb)
            ubs.append(spec.ub)

            if spec.cache:
                self.cached_indices[spec.name] = i

        # --- Step B: Process Physical Parameters (P) ---
        phy_specs_list: List[PhysicalParamSpec] = []

        if self.transform_func:
            # === Transform Mode ===
            physical_path = getattr(self.cfg.parameters, "physical", None)
            if not physical_path:
                raise ValueError("Transform mode requires cfg.parameters.physical")

            phy_raw_data = self._load_yaml(Path(physical_path))
            phy_list_raw = phy_raw_data.get("physical_parameters", [])

            seen_phy_names = set()
            for i, item in enumerate(phy_list_raw):
                spec = PhysicalParamSpec.from_dict(item, i)
                if spec.name in seen_phy_names:
                    raise ValueError(f"Duplicate physical parameter name: {spec.name}")
                seen_phy_names.add(spec.name)
                phy_specs_list.append(spec)

        else:
            # === Direct Mode ===
            # Reuse design parameters as physical parameters
            seen_phy_names = set()
            for i, item in enumerate(design_list_raw):
                spec = PhysicalParamSpec.from_dict(item, i)
                if spec.name in seen_phy_names:
                    raise ValueError(f"Duplicate physical parameter name in direct mode: {spec.name}")
                seen_phy_names.add(spec.name)
                phy_specs_list.append(spec)

        # --- Step C: Register Write Tasks ---
        self._register_write_tasks(phy_specs_list)

        return len(names), names, types, sets, ubs, lbs

    # -------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------
    def _register_write_tasks(self, phy_specs: List[PhysicalParamSpec]) -> None:
        """
        Helper method to register file writing tasks from a list of Physical Specs.
        """
        project_root = Path(self.cfg.basic.projectPath)

        for spec in phy_specs:
            if spec.cache:
                self.cached_indices_phy[spec.name] = spec.index

            # Validation
            if spec.name not in self.library:
                msg = f"Physical parameter '{spec.name}' not found in library"
                if not self.transform_func:
                    msg += " (in direct mode, design parameter names must match library names)"
                raise ValueError(msg)

            lib_info = self.library[spec.name]
            real_files = self._resolve_filenames(project_root, lib_info.file.name)

            for rel_file in real_files:
                if rel_file not in self.writeInTask:
                    abs_file = project_root / rel_file
                    self.writeInTask[rel_file] = {
                        "handler": WriteInHandler(str(abs_file)),
                        "indices": [],
                    }

                task = self.writeInTask[rel_file]
                handler: WriteInHandler = task["handler"]

                # Register param with Handler
                handler.register_param(spec, lib_info, self.cfg.parameters.hardBound)

                # Append the data source index
                task["indices"].append(spec.index)

    # -------------------------------------------------------------
    # Write Values
    # -------------------------------------------------------------
    def set_values(self, workPath: str, X, env) -> None:
        """
        Writes values to files.
        X: The optimization vector.
        """
        X_flat = np.ravel(np.asarray(X, dtype=float))

        # 1. Prepare Data Source
        try:
            if self.transform_func:
                kwargs = {"X": X_flat}
                data_source = self.func_manager.call(self.transform_func, **kwargs)

                if data_source is None:
                    raise RunError(
                        stage="subprocess",
                        code="TRANSFORM_FUNC_ERROR",
                        target="transformation",
                        message="Transform function returned None."
                    )
                if isinstance(data_source, dict):
                    raise RunError(
                        stage="subprocess",
                        code="TRANSFORM_FUNC_ERROR",
                        target="transformation",
                        message="Transform function returned a dict, but configuration expects an array-like result."
                    )

                data_source = np.ravel(np.asarray(data_source, dtype=float))
            else:
                data_source = X_flat
                
        except Exception as e:
            raise RunError(
                stage="subprocess",
                code="TRANSFORM_FUNC_EXCEPTION",
                target="transformation",
                message=f"Error during transformation: {str(e)}"
            )

        # 2. Execute Write Tasks
        work_root = Path(workPath)
        all_clamp_events: List[dict] = []

        for file_name, task in self.writeInTask.items():
            handler: WriteInHandler = task["handler"]
            target_file = work_root / file_name
            indices = task["indices"]

            try:
                if len(data_source) > 0 and len(indices) > 0 and max(indices) >= len(data_source):
                    raise RunError(
                        stage="subprocess",
                        code="INDEX_OUT_OF_BOUNDS",
                        target=file_name,
                        message=f"Data source has {len(data_source)} items, but requested index {max(indices)} "
                                f"for file '{file_name}'."
                    )

                values_to_write = data_source[indices]
                clamp_events = handler.set_values_and_save(
                    str(target_file),
                    indices,
                    values_to_write,
                )
                
                for ev in clamp_events:
                    ev["batch_id"] = env.get("batch_id", -1)
                    ev["run_id"] = env.get("i", -1)
                
                all_clamp_events.extend(clamp_events)

            except Exception as e:
                raise RunError(
                    stage="subprocess",
                    code="FILE_WRITE_ERROR",
                    target=file_name,
                    message=f"Error writing to file {file_name}: {str(e)}"
                )

        self._flush_warnings(all_clamp_events)
        
    def _flush_warnings(self, events: List[dict], warn_detail_limit: int = 20) -> None:
        """
        Writes warnings to warning.txt with timestamp, batch_id, run_id, and param info.
        Each event in 'events' is expected to contain 'file', 'param', 'idx', 'raw', 'clamped', 'lb', 'ub',
        and optionally 'batch_id' and 'run_id'.
        """
        if not events or not self.backup_path:
            return

        warning_file = self.backup_path / "warning.txt"

        head = events[:warn_detail_limit]
        msg_lines = []

        for ev in head:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            batch_id = ev.get("batch_id", -1)
            run_id = ev.get("run_id", -1)

            msg_line = (
                f"[{timestamp}] "
                f"batch={batch_id} "
                f"run={run_id} "
                f"level=WARNING "
                f"file={ev['file']}, param={ev['param']}, idx={ev['idx']}, "
                f"raw={ev['raw']:.6g}, clamped={ev['clamped']:.6g}, "
                f"bounds=[{ev['lb']}, {ev['ub']}]"
            )
            msg_lines.append(msg_line)

        more = ""
        if len(events) > warn_detail_limit:
            more = f"\n... {len(events) - warn_detail_limit} more"

        full_msg = "\n".join(msg_lines) + more

        with open(warning_file, "a", encoding="utf-8") as wf:
            wf.write(full_msg + "\n")

    # -------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------
    def get_physical_values(self, X):
        """
        Return the physical values array.
        In transform mode, this is P = f(X).
        In direct mode, this is just X.
        """
        if self.transform_func:
            kwargs = {"X": np.ravel(np.asarray(X, dtype=float))}
            result = self.func_manager.call(self.transform_func, **kwargs)
            if result is None:
                raise TypeError("Transform function returned None.")
            if isinstance(result, dict):
                raise TypeError("Transform function returned a dict, but an array-like result is required.")
            return np.ravel(np.asarray(result, dtype=float))

        return np.ravel(np.asarray(X, dtype=float))

    def get_real_X(self, X):
        """
        Backward-compatible wrapper.
        """
        return self.get_physical_values(X)

    def get_cached_env(self, X) -> Dict[str, float]:
        X_flat = np.ravel(np.asarray(X, dtype=float))

        # 1. Cache Design Variables
        env = {name: float(X_flat[i]) for name, i in self.cached_indices.items()}

        # 2. Cache Physical Parameters
        if self.cached_indices_phy:
            if self.transform_func:
                kwargs = {"X": X_flat}
                X_transform = self.func_manager.call(self.transform_func, **kwargs)

                if X_transform is None:
                    raise TypeError("Transform function returned None.")
                if isinstance(X_transform, dict):
                    raise TypeError("Transform function returned a dict, but an array-like result is required.")

                data_source = np.ravel(np.asarray(X_transform, dtype=float))
            else:
                data_source = X_flat

            env.update({name: float(data_source[i]) for name, i in self.cached_indices_phy.items()})

        return env

    def get_param_info(self):
        return self.nInput, self.xLabels, self.varType, self.varSet, self.ub, self.lb

    def _resolve_filenames(self, project_path: Union[str, Path], file_pattern: str) -> List[str]:
        """
        Supports:
        1. exact file path
        2. glob patterns: *, ?, []
        3. regex patterns with prefix 'regex:'
           Example:
             regex:^case_\\d+/input\\.dat$
        Matching is done against the POSIX-style relative path under project_path.
        """
        root = Path(project_path)

        # 1. regex mode
        if file_pattern.startswith("regex:"):
            pattern = file_pattern[len("regex:"):].strip()
            try:
                regex = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{file_pattern}': {e}") from e

            matches: List[str] = []
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(root).as_posix()
                if regex.fullmatch(rel):
                    matches.append(rel)

            matches.sort()
            if not matches:
                raise FileNotFoundError(f"Regex pattern '{file_pattern}' matched 0 files under {root}")
            return matches

        # 2. glob mode
        if any(ch in file_pattern for ch in ["*", "?", "["]):
            matches = sorted(p for p in root.glob(file_pattern) if p.is_file())
            if not matches:
                raise FileNotFoundError(f"Glob pattern '{file_pattern}' matched 0 files under {root}")
            return [p.relative_to(root).as_posix() for p in matches]

        # 3. exact file mode
        p = root / file_pattern
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"Path is not a file: {p}")
        return [p.relative_to(root).as_posix()]
