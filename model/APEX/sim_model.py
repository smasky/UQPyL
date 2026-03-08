import os
import subprocess
import shutil
import numpy as np
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from UQPyL.problem import Problem

from param_manager import ParamManager
from series_extractor import SeriesExtractor
from evaluator import Evaluator
from function_manager import FunctionManager
from load_cfg_general import load_config
from run_reporter import RunReporter

class SimModel(Problem):
    def __init__(self, cfgPath: str):
        
        self.cfgPath = cfgPath
        self.cfg = load_config(cfgPath)
        
        # Create run queue
        self.create_run_queue()
        
        # Function manager
        self.functionManager = FunctionManager(self.cfg)
        
        # Parameter manager
        self.paramManager = ParamManager(self.cfg, self.functionManager, self.backupPath)
        
        # Series
        self.seriesExtractor = SeriesExtractor(self.cfg, self.functionManager)
        
        # Objectives & diagnostics
        self.evaluator = Evaluator(self.cfg, self.functionManager)
        
        # For UQPyL
        nInput, xLabels, varType, varSet, ub, lb = self.paramManager.get_param_info()
        nOutput, optType, nConstraints = self.evaluator.get_evaluation_info()
        
        self.reporter = RunReporter(self.backupPath, xLabels, self.cfg)
        self.reporter.start()
        
        self._closed = False # for reporter
        
        super().__init__(nInput = nInput, nOutput = nOutput, nCons=nConstraints,
                         varType = varType, varSet = varSet,  
                         ub = ub, lb = lb, 
                         xLabels = xLabels,  optType = optType)

    def create_run_queue(self):
        
        nowTime = datetime.now().strftime("%m%d_%H%M")
        self.runPath = os.path.join(self.cfg.basic.workPath, "tempRun", nowTime)
        
        if os.path.exists(self.runPath):
            self.runPath = self.runPath + f"_{np.random.randint(100)}"
        os.makedirs(self.runPath, exist_ok=True)
            
        self.runQueue = queue.Queue()
        
        for i in range(self.cfg.basic.parallel):
            path = os.path.join(self.runPath, f"instance_{i}")
            shutil.copytree(self.cfg.basic.projectPath, path)
            self.runQueue.put(path)
        
        self.backupPath = os.path.join(self.runPath, "backup")
        
        os.makedirs(self.backupPath)
        
        shutil.copy(self.cfg.parameters.library, self.backupPath)
        shutil.copy(self.cfg.parameters.params, self.backupPath)
        shutil.copy(self.cfgPath, self.backupPath)
        
        for series in self.cfg.series:
            if series.obs:
                shutil.copy(os.path.join(self.cfg.basic.workPath, series.obs.file), self.backupPath)

    def evaluate(self, X):
        n = X.shape[0]
        objs = np.zeros((n, self.evaluator.nOutput))
        batch_id = self.reporter.new_batch_id()
        
        records = []
        
        cons = np.zeros((n, self.evaluator.nConstraints)) if self.evaluator.nConstraints > 0 else None
        
        if self.cfg.basic.parallel > 1:
            with ThreadPoolExecutor(max_workers=self.cfg.basic.parallel) as executor:
                futures = [executor.submit(self._subprocess, X[i, :], i, batch_id) for i in range(n)]
                records = [future.result() for future in futures]
        else:
            for i in range(n):
                context = self._subprocess(X[i, :], i, batch_id)
                records.append(context)
        
        # TODO: handle constraints
        for cte in records:
            i = cte['i']
            if 'error' in cte:
                objs[i, :] = -1 * np.inf #TODO
            else:
                for j, obj_id in enumerate(self.cfg.objectives.use):
                    if np.isnan(cte[obj_id]):
                        objs[i, j] = np.inf * self.opt[j]
                    else:
                        objs[i, j] = cte[obj_id]
                
                for j, con_id in enumerate(self.cfg.constraints.use):
                    if np.isnan(cte[con_id]):
                        cons[i, j] = -1*np.inf
                    else:
                        cons[i, j] = cte[con_id]
                
        return {"objs": objs, "cons": cons}
    
    def objFunc(self, X):
        res = self.evaluate(X)
        return res['objs']
        
    def conFunc(self, X):
        res = self.evaluate(X)
        return res['cons']
    
    def _subprocess(self, X, i, batch_id):
        workPath = self.runQueue.get()
        
        context = {'X': X.ravel(), 'i': i, 'batch_id': batch_id}
        
        try:
            self.paramManager.set_values(workPath, X)
            context.update(self.paramManager.get_cached_env(X))
            
            process = subprocess.Popen(
                os.path.join(workPath, self.cfg.basic.command),
                cwd=workPath,
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            t = self.cfg.basic.timeout if self.cfg.basic.timeout > 0 else None
            
            # TODO
            try:
                stdout, stderr = process.communicate(timeout = t)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            
            context = self.seriesExtractor.extract_all(workPath, context)
            
            scalars = self.evaluator.evaluate_all(context)
            
            context.update(scalars)
            
        except Exception as e:
            context['error'] = str(e)
            print(f"[Instance {i} Failed]: {e}")
        
        finally:
            self.runQueue.put(workPath)
            if hasattr(self, "reporter"):
                self.reporter.submit(context)
            
        return context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def close(self):
        if getattr(self, "_closed", False):
            return
        
        self._closed = True
        
        if hasattr(self, "reporter") and self.reporter is not None:
            try:
                self.reporter.close()
            except Exception as e:
                print(f"[SimModel.close] reporter.close() failed: {e}")
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
                              