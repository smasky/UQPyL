import numpy as np
import sys
import inspect
import importlib.util
# built-in metrics
from UQPyL.util.metric import r_square, mse, nse, rank_score, sort_score

BUILTIN_FUNCS = {
    "R2": lambda sim, obs: r_square(obs, sim),
    "RMSE": lambda sim, obs: np.sqrt(mse(obs, sim)),
    "MSE": lambda sim, obs: mse(obs, sim),
}

class Evaluator:
    def __init__(self, cfg, funcManager):
        self.cfg = cfg
        self.projectPath = self.cfg.basic.projectPath
        
        self.funcManager = funcManager
        
        # self.functions = self._register_functions()

        self.nOutput, self.optType, self.obj_refs = self._parse_objectives()
        
        self.nOutput, self.optType, self.obj_refs = self._parse_objectives()
        self.diag_refs = self._parse_diagnostics()
    
    def _parse_objectives(self):
        
        optType = []
        obj_refs = {}
        
        for obj_id in self.cfg.objectives.use:
            
            obj_cfg = self.cfg.objectives.items[obj_id]
            
            #TODO
            optType.append(obj_cfg.sense)
            obj_refs[obj_id] = obj_cfg.ref
        
        nOutput = len(optType)
        
        return nOutput, optType, obj_refs
    
    def _parse_diagnostics(self):
        
        diag_refs = {}
        
        for diag_id in self.cfg.diagnostics.use:
            
            diag_cfg = self.cfg.diagnostics.items[diag_id]
            
            diag_refs[diag_id] = diag_cfg.ref
        
        return diag_refs
    
    def evaluate_all(self, context):
        
        record = {}
        
        for derived in self.cfg.derived:
            d_id = derived.id
            func_name = derived.call.func
            args_map = derived.call.args
            
            kwargs = {}
            for arg_name, context_key in args_map.items():
                if context_key not in context:
                    raise KeyError(f"Derived '{d_id}' requires context key '{context_key}'")
                
                val = context[context_key]
                if isinstance(val, list):
                    val = np.array(val).ravel()
                kwargs[arg_name] = val
                
            result = self.funcManager.call(func_name, **kwargs)
            context[d_id] = result
            
        for obj_id, ref_id in self.obj_refs.items():
            if ref_id not in context:
                raise KeyError(f"Objective '{obj_id}' requires context key '{ref_id}'")
            
            val = float(context[ref_id])
            record[obj_id] = val
        
        for diag_id, ref_id in self.diag_refs.items():
            if ref_id not in context:
                raise KeyError(f"Diagnostic '{diag_id}' requires context key '{ref_id}'")
                
            val = float(context[ref_id])
            record[diag_id] = val

        return record


    def get_evaluation_info(self):
        
        return self.nOutput, self.optType