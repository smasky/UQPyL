import re
from extract_handler import read_extract
from load_cfg_general import ExprSpec, CallSpec

class SeriesExtractor:
    def __init__(self, cfg, funcManager):
        
        self.cfg = cfg
        
        self.funcManager = funcManager
        
        required_sids = self._parse_required_sids()
        
        self.seriesDict = self._init_series(required_sids)
        
    def _parse_required_sids(self) -> set:
        required_sids = set()
        for d in self.cfg.derived:
            for k, v in d.call.args.items():
                if isinstance(v, str) and v.endswith("_sim"):
                    required_sids.add(v.replace("_sim", ""))
        
        for s in self.cfg.series:
            if getattr(s, 'cache', False):
                required_sids.add(s.id)

        return required_sids
    def _init_series(self, cache_sids: set) -> dict:
        
        seriesDict = {}
        
        for s in self.cfg.series:
            
            if s.id not in cache_sids:
                continue

            series_id = s.id
            obsItem = s.obs
            simItem = s.sim
            
            obsData = None
            if obsItem:
                obsData = read_extract(None, obsItem)

            seriesDict[series_id] = {
                'obs': obsData, 
                'simItem': simItem, 
                'obsItem': obsItem,
            }
        
        return seriesDict
    
    def extract_all(self, workPath: str, context: dict) -> dict:
        
        env = context.copy()
        
        for sid, item in self.seriesDict.items():
            simItem = item['simItem']
            
            if isinstance(simItem, ExprSpec):
                for dep in simItem.deps:
                    if dep not in env:
                        is_obs = dep.endswith("_obs")
                        base_id = dep.replace("_sim", "").replace("_obs", "")
                        
                        if is_obs:
                            obs_data = self.seriesDict[base_id]['obs']
                            if obs_data is None:
                                raise ValueError(f"Expr dependency error: formula requested '{dep}' but '{base_id}' has no obs file")
                            env[dep] = obs_data
                        else:
                            si = self.cfg.series_index[base_id].sim
                            val = read_extract(workPath, si)
 
                            env[f"{base_id}_sim"] = val
                try:
                    env[f"{sid}_sim"] = eval(simItem.expr, {"__builtins__": {}}, env)
                    
                    if item['obs'] is not None:
                        env[f"{sid}_obs"] = item['obs']
                    
                except NameError as e:
                    raise ValueError(f"Expr evaluation failed for series '{sid}'. Missing dependency: {e}")
            elif isinstance(simItem, CallSpec):

                func_name = simItem.func
                raw_args = simItem.args
                
                func_args = {}
                for arg_k, arg_v in raw_args.items():
                    if arg_v not in env:
                        is_obs = arg_v.endswith("_obs")
                        base_id = arg_v.replace("_sim", "").replace("_obs", "")
                        if is_obs:
                            obs_data = self.seriesDict[base_id]['obs']
                            if obs_data is None:
                                raise ValueError(f"Expr dependency error: formula requested '{arg_v}' but '{base_id}' has no obs file")
                            env[arg_v] = obs_data
                        else:
                            si = self.cfg.series_index[base_id].sim
                            val = read_extract(workPath, si)

                            env[arg_v] = val
                    func_args[arg_k] = env[arg_v]
                    
                func_result = self.funcManager.call(func_name, **func_args)
                
                env[f"{sid}_sim"] = func_result
                
                if item['obs'] is not None:
                        env[f"{sid}_obs"] = item['obs']
            else:
                sim = f"{sid}_sim"
                obs = f"{sid}_obs"
                
                if sim not in env:
                    val = read_extract(workPath, simItem)
                    env[sim] = val
                
                if obs not in env:
                    if item['obs'] is not None:
                        env[obs] = item['obs']
        
        context.update(env)
        
        return context