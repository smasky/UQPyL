import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

# built-in metrics
from UQPyL.util.metric import r_square, mse


BUILTIN_FUNCS: Dict[str, Callable[..., Any]] = {
    "R2": lambda sim, obs: r_square(obs, sim),
    "RMSE": lambda sim, obs: np.sqrt(mse(obs, sim)),
    "MSE": lambda sim, obs: mse(obs, sim),
}


class FunctionManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.functions: Dict[str, Callable[..., Any]] = {}
        self._module_cache: Dict[Path, Any] = {}
        self._load_functions()

    def _load_functions(self) -> None:
        # Load all functions declared in cfg.functions
        for alias, f_spec in self.cfg.functions.items():
            if f_spec.kind == "builtin":
                self.functions[alias] = self._load_builtin_func(alias, f_spec.name)
            elif f_spec.kind == "external":
                self.functions[alias] = self._load_external_func(
                    alias=alias,
                    func_name=f_spec.name,
                    file_path=f_spec.file,
                )
            else:
                raise ValueError(f"Unsupported function kind: {f_spec.kind}")

            self._validate_declared_args(alias, self.functions[alias], f_spec.args)

    def _load_builtin_func(self, alias: str, builtin_name: str) -> Callable[..., Any]:
        # Resolve builtin function by declared builtin name
        if builtin_name not in BUILTIN_FUNCS:
            available = ", ".join(sorted(BUILTIN_FUNCS))
            raise ValueError(
                f"Unknown builtin function '{builtin_name}' for alias '{alias}'. "
                f"Available builtins: {available}"
            )
        return BUILTIN_FUNCS[builtin_name]

    def _load_external_func(self, alias: str, func_name: str, file_path: Path) -> Callable[..., Any]:
        # Load external function from a Python file
        if file_path is None:
            raise ValueError(f"External function '{alias}' requires a file path.")

        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"External function file not found: {file_path}")

        module = self._load_module_from_file(file_path)

        if not hasattr(module, func_name):
            raise ValueError(
                f"Function '{func_name}' not found in external file: {file_path}"
            )

        func = getattr(module, func_name)

        if not callable(func):
            raise TypeError(
                f"Attribute '{func_name}' in {file_path} exists but is not callable."
            )

        return func

    def _load_module_from_file(self, file_path: Path):
        # Load and cache a Python module from file
        if file_path in self._module_cache:
            return self._module_cache[file_path]

        unique_name = self._build_module_name(file_path)
        spec = importlib.util.spec_from_file_location(unique_name, str(file_path))

        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec from file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)

        self._module_cache[file_path] = module
        return module

    @staticmethod
    def _build_module_name(file_path: Path) -> str:
        # Build a unique module name to avoid sys.modules collisions
        digest = hashlib.md5(str(file_path).encode("utf-8")).hexdigest()[:10]
        return f"_uq_func_{file_path.stem}_{digest}"

    @staticmethod
    def _validate_declared_args(alias: str, func: Callable[..., Any], declared_args) -> None:
        # Validate that declared args are compatible with the Python callable
        if not declared_args:
            return

        sig = inspect.signature(func)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if has_var_kw:
            return

        missing = [arg for arg in declared_args if arg not in params]
        if missing:
            raise ValueError(
                f"Function '{alias}' declared args {missing}, "
                f"but callable signature is {sig}"
            )

    def has_function(self, func_name: str) -> bool:
        # Check whether a function is registered
        return func_name in self.functions

    def get(self, func_name: str) -> Callable[..., Any]:
        # Return a registered function
        if func_name not in self.functions:
            available = ", ".join(sorted(self.functions))
            raise ValueError(
                f"Function '{func_name}' is not registered. "
                f"Available functions: {available}"
            )
        return self.functions[func_name]

    def call(self, func_name: str, **kwargs):
        # Call a registered function with keyword arguments
        func = self.get(func_name)
        try:
            return func(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed calling function '{func_name}' with kwargs={list(kwargs.keys())}: {e}"
            ) from e
