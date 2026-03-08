import os
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from write_in_handler import WriteInHandler

# -------------------------------------------------------------
# Constant Mappings
# -------------------------------------------------------------
TYPE_MAP = {"float": 0, "int": 1, "discrete": 2}
MODE_MAP = {"r": 0, "v": 1, "a": 2} # r: relative, v: value (replace), a: add

# -------------------------------------------------------------
# Data Classes
# -------------------------------------------------------------

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
        
        raw_line = d["file"]["line"]
        if isinstance(raw_line, list) and len(raw_line) > 0 and isinstance(raw_line[0], list):
            parsed_line = _expand_row_ranges(raw_line)
        else:
            parsed_line = raw_line
        
        return ParamSpec(
            name=d["name"],
            type=TYPE_MAP.get(d["type"], 0),
            bounds=d["bounds"],
            file=ParamFileSpec(name=d["file"]["name"], line=parsed_line, 
                               start=d["file"]["start"], width=d["file"]["width"], 
                               precision=d["file"]["precision"],
                               maxNum=int(d["file"].get("maxNum", 1))),
        )
    
    @property
    def lb(self) -> float: return float(self.bounds[0])
    @property
    def ub(self) -> float: return float(self.bounds[1])

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
    def from_dict(d: Dict[str, Any], index: int):
        p_name = d.get("name")
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
            cache=d.get("cache", False),
        )
    
    @property
    def lb(self) -> float: return float(self.bounds[0])
    @property
    def ub(self) -> float: return float(self.bounds[1])

@dataclass
class PhysicalParamSpec:
    """Represents a Physical Parameter (P) that will be written to files."""
    index: int      # Index in the source array (P array or X array)
    name: str       # Name in the Library
    mode: int       # r/v/a
    cache: bool = False
    
    @staticmethod
    def from_dict(d: Dict[str, Any], index: int):
        return PhysicalParamSpec(
            index=index,
            name=d["name"],
            mode=MODE_MAP.get(d.get("mode", "v"), 1), # Default to 'v' (replace)
            cache=d.get("cache", False)
        )

# -------------------------------------------------------------
# Parameter Manager
# -------------------------------------------------------------

class ParamManager:
    def __init__(self, cfg, functionManager, backupPath):
        
        self.cfg = cfg
        self.func_manager = functionManager
        
        self.backupPath = backupPath
        
        self.writeInTask = {}         
        self.cached_indices = {}      # Cache for X
        self.cached_indices_phy = {}  # Cache for P
        
        # 1. Read Library
        self.library: Dict[str, ParamSpec] = self._read_library()
        
        # 2. Check for Transform Function
        self.transform_func = None
        if self.cfg.parameters.func:
             self.transform_func = self.cfg.parameters.func

        # 3. Unified Initialization
        # Returns optimization metadata needed by the optimizer
        self.nInput, self.xLabels, self.varType, self.varSet, self.ub, self.lb = self._init_params()
        
    def _read_library(self):
        lib_path = self.cfg.parameters.library
        with open(lib_path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)
        
        library_dict = {}
        for item in raw_data.get("parameter_library", []):
            spec = ParamSpec.from_dict(item)
            library_dict[spec.name] = spec
        return library_dict
    
    def _init_params(self):
        """
        Unified logic to initialize Design Variables (X) and Physical Parameters (P).
        """
        
        # --- Step A: Process Design Variables (X) ---
        # Always reads from 'params' (or 'design_variables') file
        with open(self.cfg.parameters.params, 'r', encoding='utf-8') as f:
            design_raw_data = yaml.safe_load(f)
            
        design_list_raw = design_raw_data.get("design_parameters", [])
        
        names, types = [], []
        ubs, lbs, sets = [], [], {}
        
        for i, item in enumerate(design_list_raw):
            # Create Design Spec
            spec = DesignParamSpec.from_dict(item, i)
            
            names.append(spec.name)
            types.append(spec.type)
            if spec.type == 2: sets[i] = spec.sets
            lbs.append(spec.lb)
            ubs.append(spec.ub)
            
            if spec.cache:
                self.cached_indices[spec.name] = i

        # --- Step B: Process Physical Parameters (P) ---
        # Strategy: Determine the list of Physical Specs based on mode
        
        phy_specs_list = []
        
        if self.transform_func:
            # === Transform Mode ===
            # P is defined explicitly in a separate physical parameters file
            with open(self.cfg.parameters.physical, 'r', encoding='utf-8') as f:
                phy_raw_data = yaml.safe_load(f)
            
            phy_list_raw = phy_raw_data.get("physical_parameters", [])
            for i, item in enumerate(phy_list_raw):
                phy_specs_list.append(PhysicalParamSpec.from_dict(item, i))
                
        else:
            # === Direct Mode ===
            # P is implied from X (Design Variables)
            # We reuse the raw data from Step A, but treat them as Physical Specs
            for i, item in enumerate(design_list_raw):
                # In Direct Mode, X maps 1:1 to P, so index 'i' matches
                phy_specs_list.append(PhysicalParamSpec.from_dict(item, i))

        # --- Step C: Register Write Tasks ---
        # Valid for both modes
        self._register_write_tasks(phy_specs_list)

        return len(names), names, types, sets, ubs, lbs

    def _register_write_tasks(self, phy_specs: List[PhysicalParamSpec]):
        """
        Helper method to register file writing tasks from a list of Physical Specs.
        """
        for spec in phy_specs:
            
            if spec.cache:
                self.cached_indices_phy[spec.name] = spec.index
            
            # Validation
            if spec.name not in self.library:
                msg = f"Physical parameter '{spec.name}' not found in Library!"
                if not self.transform_func:
                    msg += " (In Direct Mode, Design Variables must match Library names)"
                raise ValueError(msg)
            
            lib_info = self.library[spec.name]
            real_files = self._resolve_filenames(self.cfg.basic.projectPath, lib_info.file.name)
            
            for rf in real_files:
                if rf not in self.writeInTask:
                    self.writeInTask[rf] = {
                        "handler": WriteInHandler(os.path.join(self.cfg.basic.projectPath, rf)),
                        "indices": [],
                    }
                
                task = self.writeInTask[rf]
                handler = task["handler"]
                
                # Register param with Handler
                handler.register_param(spec, lib_info, self.cfg.parameters.hardBound)
                
                # Append the data source index (either index in X or index in P array)
                task["indices"].append(spec.index)

    def set_values(self, workPath: str, X):
        """
        Writes values to files.
        X: The optimization vector.
        """
        X_flat = np.ravel(X)
        
        # 1. Prepare Data Source
        if self.transform_func:
            # === Transform Mode ===
            # Calculate P array from X
            kwargs = {'X': X_flat}
            data_source = self.func_manager.call(self.transform_func, **kwargs)
            
            if isinstance(data_source, dict):
                raise TypeError("Transform function returned a dict, but configuration expects an array.")
            data_source = np.ravel(data_source)
        else:
            # === Direct Mode ===
            # X is the source directly
            data_source = X_flat

        # 2. Execute Write Tasks
        for file_name, task in self.writeInTask.items():
            handler = task["handler"]
            target_file = os.path.join(workPath, file_name)
            indices = task["indices"]
            
            # Safety check
            if len(data_source) > 0 and len(indices) > 0:
                 if max(indices) >= len(data_source):
                     raise ValueError(f"Data source has {len(data_source)} items, but requested index {max(indices)}.")

            values_to_write = data_source[indices]
            handler.set_values_and_save(target_file, indices, values_to_write, backup_path=self.backupPath)

    # -------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------
    
    def get_real_X(self, X):
        if self.transform_func:
             kwargs = {'X': np.ravel(X)}
             return self.func_manager.call(self.transform_func, **kwargs)
        return X

    def get_cached_env(self, X):
        X_flat = np.ravel(X)
        
        # 1. Cache Design Variables
        env = {name: float(X_flat[i]) for name, i in self.cached_indices.items()}
        
        # 2. Cache Physical Parameters (Only if needed)
        if self.cached_indices_phy:
            # We need to calculate P if we want to cache P
            if self.transform_func:
                kwargs = {'X': X_flat}
                X_transform = self.func_manager.call(self.transform_func, **kwargs)
                data_source = np.ravel(X_transform)
            else:
                data_source = X_flat
            
            env.update({name: float(data_source[i]) for name, i in self.cached_indices_phy.items()})
        
        return env

    def get_param_info(self):
        return self.nInput, self.xLabels, self.varType, self.varSet, self.ub, self.lb
    
    def _resolve_filenames(self, project_path: str, file_pattern: str) -> List[str]:
        root = Path(project_path)
        if any(ch in file_pattern for ch in ["*", "?", "["]):
            matches = sorted(root.glob(file_pattern))
            if not matches:
                raise FileNotFoundError(f"Pattern {file_pattern} matched 0 files under {root}")
            return [str(p.relative_to(root).as_posix()) for p in matches]
        else:
            p = root / file_pattern
            if not p.exists():
                raise FileNotFoundError(f"File {p} not found")
            return [file_pattern]
