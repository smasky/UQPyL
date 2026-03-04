import importlib.util
import os
import sys
from typing import Dict, Any, Callable
import numpy as np

# built-in metrics
from UQPyL.util.metric import r_square, mse, nse, rank_score, sort_score

BUILTIN_FUNCS = {
    "R2": lambda sim, obs: r_square(obs, sim),
    "RMSE": lambda sim, obs: np.sqrt(mse(obs, sim)),
    "MSE": lambda sim, obs: mse(obs, sim),
}

class FunctionManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.functions: Dict[str, Callable] = {}
        self._load_functions()

    def _load_functions(self):
    
        for name, f_spec in self.cfg.functions.items():
            if f_spec.kind == "builtin":
                
                if f_spec.name in BUILTIN_FUNCS:
                    self.functions[name] = BUILTIN_FUNCS[f_spec.name]
                else:
                    raise ValueError(f"Unknown builtin function: {f_spec.name}")
            
            elif f_spec.kind == "external":
                
                self._load_external_func(name, f_spec.file)

    def _load_external_func(self, func_name: str, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"External function file not found: {file_path}")
        
        spec = importlib.util.spec_from_file_location("external_module", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["external_module"] = module
        spec.loader.exec_module(module)
        
        if hasattr(module, func_name):
            self.functions[func_name] = getattr(module, func_name)
        else:
            raise ValueError(f"Function '{func_name}' not found in {file_path}")

    def call(self, func_name: str, **kwargs):
       
        if func_name not in self.functions:
            raise ValueError(f"Function '{func_name}' is not registered.")
        return self.functions[func_name](**kwargs)
