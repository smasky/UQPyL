import numpy as np
from load_cfg_general import expand_row_ranges
from run_error import RunError


class Evaluator:
   def __init__(self, cfg, funcManager):
       self.cfg = cfg
       self.funcManager = funcManager

       self.nOutput, self.optType, self.obj_refs, self.nConstraints, self.con_refs = (
           self._parse_objectives_constraints()
       )
       self.diag_refs = self._parse_diagnostics()

   def _parse_objectives_constraints(self):
       optType = []
       obj_refs = {}
       con_refs = {}

       for obj_id in self.cfg.objectives.use:
           obj_cfg = self.cfg.objectives.items[obj_id]
           optType.append(obj_cfg.sense)
           obj_refs[obj_id] = obj_cfg.ref

       nOutput = len(optType)

       for con_id in self.cfg.constraints.use:
           con_cfg = self.cfg.constraints.items[con_id]
           con_refs[con_id] = con_cfg.ref

       nConstraints = len(con_refs)

       return nOutput, optType, obj_refs, nConstraints, con_refs

   def _parse_diagnostics(self):
       diag_refs = {}

       for diag_id in self.cfg.diagnostics.use:
           diag_cfg = self.cfg.diagnostics.items[diag_id]
           diag_refs[diag_id] = diag_cfg.ref

       return diag_refs

   def _normalize_value(self, val):
       if isinstance(val, (list, tuple, np.ndarray)):
           return np.asarray(val).ravel()
       return val

   def _to_scalar(self, value, label: str) -> float:
       value = self._normalize_value(value)

       if isinstance(value, np.ndarray):
           if value.size != 1:
               raise ValueError(
                   f"{label} must be scalar, but got array with shape {value.shape}"
               )
           return float(value.item())

       return float(value)

   def _collect_record_values(self, refs: dict, env: dict, kind: str) -> dict:
       result = {}

       for item_id, ref_id in refs.items():
           if ref_id not in env:
               raise RunError(
                   stage="evaluator",
                   code="MISSING_CONTEXT",
                   target=item_id,
                   message=f"{kind} '{item_id}' requires context key '{ref_id}'"
               )

           try:
               result[item_id] = self._to_scalar(env[ref_id], f"{kind} '{item_id}'")
           except Exception as e:
               raise RunError(
                   stage="evaluator",
                   code="INVALID_VALUE",
                   target=item_id,
                   message=f"{kind} '{item_id}' cannot be converted to scalar: {e}"
               ) from e

       return result

   def evaluate_all(self, context):
       env = context
       record = {}

       for derived in self.cfg.derived:
           d_id = derived.id

           if getattr(derived, "call", None):
               func_name = derived.call.func
               args_map = derived.call.args
               kwargs = {}

               for arg_name, context_key in args_map.items():
                   if context_key not in env:
                       raise RunError(
                           stage="derived",
                           code="DEPENDENCY_MISSING",
                           target=d_id,
                           message=f"Derived '{d_id}' requires context key '{context_key}'"
                       )

                   kwargs[arg_name] = self._normalize_value(env[context_key])

               try:
                   result = self.funcManager.call(func_name, **kwargs)
               except Exception as e:
                   raise RunError(
                       stage="derived",
                       code="FUNC_CALL_FAILED",
                       target=d_id,
                       message=f"Function '{func_name}' failed: {e}"
                   ) from e

           elif getattr(derived, "expr", None):
               deps = expand_row_ranges(derived.expr)
               expr_env = {}

               for dep in deps:
                   if dep not in env:
                       raise RunError(
                           stage="derived",
                           code="DEPENDENCY_MISSING",
                           target=d_id,
                           message=f"Derived '{d_id}' expr requires context key '{dep}'"
                       )

                   expr_env[dep] = self._normalize_value(env[dep])

               try:
                   result = eval(derived.expr, {"__builtins__": {}, "np": np}, expr_env)
               except NameError as e:
                   raise RunError(
                       stage="derived",
                       code="EXPR_EVAL_FAILED",
                       target=d_id,
                       message=f"Derived '{d_id}' expr evaluation failed: {e}"
                   ) from e
               except Exception as e:
                   raise RunError(
                       stage="derived",
                       code="EXPR_EVAL_FAILED",
                       target=d_id,
                       message=f"Derived '{d_id}' expr evaluation failed: {e}"
                   ) from e

           else:
               raise RunError(
                   stage="derived",
                   code="INVALID_CONFIG",
                   target=d_id,
                   message=f"Derived '{d_id}' has neither 'call' nor 'expr'"
               )

           if result is None:
               raise RunError(
                   stage="derived",
                   code="EMPTY_RESULT",
                   target=d_id,
                   message=f"Derived '{d_id}' returned None"
               )

           env[d_id] = result

       record.update(self._collect_record_values(self.obj_refs, env, "Objective"))
       record.update(self._collect_record_values(self.con_refs, env, "Constraint"))
       record.update(self._collect_record_values(self.diag_refs, env, "Diagnostic"))

       return record

   def get_evaluation_info(self):
       return self.nOutput, self.optType, self.nConstraints