from sys import modules

from UQPyL.problem import Problem
from UQPyL.util.metric import r_square, mse
import pandas as pd
import os
import subprocess
import shutil
import numpy as np
import queue
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from write_in_handler import WriteInHandler
from extract_handler import read_extract
from load_cfg_general import load_config
from run_reporter import RunReporter

TYPE_MAP = {"float" : 0, "int" : 1, "discrete" : 2}
MODE_MAP = {"r" : 0, "v" : 1, "a" : 2}
METRIC_MAP = {"R2": r_square, "MSE": mse}

class APEX(Problem):
    def __init__(self, cfgPath: str):
        
        self.cfg = load_config(cfgPath)
        
        # parameter
        self.read_paramInfo()
        nInput, xLabels, varType, varSet, ub, lb = self.read_param()
        
        # objectives
        nOutput, optType, objectivesDict, obj_sids_map, cache1 = self.read_objectives()
        self.objectivesDict = objectivesDict
        
        # diagnostics
        diagnosticsDict, diag_sids_map, cache2 = self.read_diagnostics()
        
        self.diagnosticsDict = diagnosticsDict
        
        # series
        cache = set(cache1 + cache2)
        self.seriesDict = self.read_series(cache)
        
        # Create run queue
        self.create_run_queue()
        
        # count for report
        self.count = 0
        
        super().__init__(nInput = nInput, nOutput = nOutput, 
                         varType = varType, varSet = varSet,  
                         ub = ub, lb = lb, 
                         xLabels = xLabels,  optType = optType, name = 'APEX')
        
        self.reporter = RunReporter(self.backupPath, xLabels, self.seriesDict.keys(), self.objectivesDict.keys(), self.diagnosticsDict.keys(), obj_sids_map, diag_sids_map, self.cfg)
        self.reporter.start()

    def create_run_queue(self):
        
        nowTime = datetime.now().strftime("%m%d_%H%M")
        self.runPath = os.path.join(self.cfg.basic.workPath, "tempRun", nowTime)
        
        if os.path.exists(self.runPath):
            os.makedirs(self.runPath + f"_{np.random.randint(100)}")
            
        self.runQueue = queue.Queue()
        
        for i in range(self.cfg.basic.parallel):
            path = os.path.join(self.runPath, f"instance_{i}")
            shutil.copytree(self.cfg.basic.projectPath, path)
            self.runQueue.put(path)
        
        self.backupPath = os.path.join(self.runPath, "backup")
        os.makedirs(self.backupPath)
        shutil.copy(self.cfg.parameters.info, self.backupPath)
        shutil.copy(self.cfg.parameters.param, self.backupPath)
        
        for series in self.cfg.series:
            if series.obs:
                shutil.copy(os.path.join(self.cfg.basic.workPath, series.obs.file), self.backupPath)

    def read_objectives(self):
        
        cache = []
        
        optType = []
        
        objectivesDict = {}
        
        obj_sids_map = {}
        
        for obj in self.cfg.objectives.use:
            
            comb = {}
            
            obj_cfg = self.cfg.objectives.items[obj]

            optType.append(obj_cfg.direction)
            
            obj_sids_map[obj] = []
            
            for term in obj_cfg.terms:
                
                cache.append(term.id)
                
                comb[term.id] = term.weight
                
                obj_sids_map[obj].append(term.id)
            
            objectivesDict[obj] = {'name': obj_cfg.name, 'comb': comb, 'reduce': obj_cfg.reduce, 'metric': METRIC_MAP[obj_cfg.metric]}

        nOutput = len(optType)
        
        return nOutput, optType, objectivesDict, obj_sids_map, cache
       
    def read_diagnostics(self):
        
        diagnosticsDict = {}
        
        diag_sids_map = {}
        
        cache = []
        for diag in self.cfg.diagnostics.use:
            
            diag_cfg = self.cfg.diagnostics.items[diag]
            
            comb = {}
            
            diag_sids_map[diag] = []
            
            for term in diag_cfg.terms:
                cache.append(term.id)
                comb[term.id] = term.weight
                diag_sids_map[diag].append(term.id)
            
            diagnosticsDict[diag] = {'name': diag_cfg.name, 'reduce': diag_cfg.reduce, 'comb': comb, 'metric': METRIC_MAP[diag_cfg.metric]}
            
        return diagnosticsDict, diag_sids_map, cache
    
    def read_series(self, cache):
        
        seriesDict = {}
        
        for sid in cache:
            
            series = self.cfg.series_index[sid]
            
            obsItem = series.obs
            simItem = series.sim
            
            if obsItem:
                obsData = read_extract(self.cfg.basic.workPath, obsItem)
            else:
                obsData = None
            
            seriesDict[sid] = {'obs': obsData, 'simItem': simItem, 'obsItem': obsItem}
        
        return seriesDict
    
    def read_paramInfo(self):
        
        try:
            self.basicInfo = pd.read_csv(
                self.cfg.parameters.info,
                sep=",",              
                comment="#",      
            )
            
            for col in ["Line", "Start", "Width", "Precision"]:
                self.basicInfo[col] = pd.to_numeric(self.basicInfo[col], errors="coerce").astype("Int64")
            
            for col in ["UB", "LB"]:
                self.basicInfo[col] = pd.to_numeric(self.basicInfo[col], errors="coerce").astype("Float64")
            
            self.basicInfo = self.basicInfo.set_index("Name")
            
        except:
            raise ValueError(f"The location file {self.cfg.parameters.info} is not valid, please check the file!")
    
    def read_param(self):
        
        def _is_glob(p: str) -> bool:
            return any(ch in p for ch in ["*", "?", "["])

        def _resolve_filenames(projectPath: str, fileName: str):
            root = Path(projectPath)
            if _is_glob(fileName):
                matches = sorted(root.glob(fileName))
                if not matches:
                    raise FileNotFoundError(f"Pattern {fileName} matched 0 files under {root}")
                return [str(p.relative_to(root)) for p in matches]
            else:
                p = root / fileName
                if not p.exists():
                    raise FileNotFoundError(f"File {p} not found")
                return [fileName]
        
        names = []; types = []; modes = []
        ubs = []; lbs = []; sets = {}
        
        with open(self.cfg.parameters.param, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines[1:]):
            contents = line.split()
            names.append(contents[0])
            
            types.append(TYPE_MAP[contents[1]])
            modes.append(MODE_MAP[contents[2]])
            
            if TYPE_MAP[contents[1]] != 2:
                lbs.append(float(contents[3].split('_')[0]))
                ubs.append(float(contents[3].split('_')[1]))
            else:
                sets[i] = [float(i) for i in contents[3].split('_')]
                lbs.append(0)
                ubs.append(1)

        #
        writeInTask = {}
        
        for i, (name, mode) in enumerate(zip(names, modes)):
            
            ub = None if not self.cfg.basic.hardBound else self.basicInfo.loc[name,'UB']
            lb = None if not self.cfg.basic.hardBound else self.basicInfo.loc[name,'LB']
            
            lineNum, start, width, precision = self.basicInfo.loc[name, ["Line", "Start", "Width", "Precision"]]
            
            fileName = self.basicInfo.loc[name, "Filename"]
            
            real_files = _resolve_filenames(self.cfg.basic.projectPath, fileName)
            
            for rf in real_files:
                
                if rf not in writeInTask:
                    writeInTask[rf] = {}
                    writeInTask[rf]["indices"] = []
                
                if "Handler" not in writeInTask[rf]:
                    handler = WriteInHandler(os.path.join(self.cfg.basic.projectPath, rf))
                    writeInTask[rf]["Handler"] = handler
                else:
                    handler = writeInTask[rf]["Handler"]
                
                handler.register_param(name, i, mode, TYPE_MAP[self.basicInfo.loc[name, "Type"]], lineNum, start, width, precision, lb, ub)
                
                writeInTask[rf]["indices"].append(i)

        self.writeInTask = writeInTask
        nInput = len(names)
        return nInput, names, types, sets, ubs, lbs
    
    def evaluate(self, X):
        
        n = X.shape[0]
        nOut = self.nOutput
        objs = np.zeros((n, nOut))
        # nCons = self.nCons
        # cons = np.zeros((n, self.nCons)) if nCons > 0 else None
        
        batch_id = self.reporter.new_batch_id()
        records = []
        
        
        if self.cfg.basic.parallel > 1:
            with ThreadPoolExecutor(max_workers = self.cfg.basic.parallel) as executor:
                futures = [executor.submit(self._subprocess, X[i, :], i, batch_id) for i in range(n)]
                records = [future.result() for future in futures]
        else:
            for i in range(n):
                r = self._subprocess(X[i, :], i, batch_id)
                records.append(r)
        
        # for i in range(1, n + 1):
        #     r = self._subprocess(X[i, :], i, batch_id)
        #     records.append(r)

        # with ThreadPoolExecutor(max_workers = self.cfg.basic.parallel) as executor:
        #     futures = [executor.submit(self._subprocess, X[i, :], i, batch_id) for i in range(n)]
        #     records = [future.result() for future in futures]
        
        for r in records:
            i = r['i']
            if 'error' in r:
                objs[i, :] = np.inf   #TODO
            else:
                for j, obj_id in enumerate(self.cfg.objectives.use):
                    objs[i, j] = r[obj_id]['agg_val']
        
        return objs

    def close(self):
        if hasattr(self, "reporter"):
            self.reporter.close()
    
    def _set_values(self, workPath, X):
        
        for fileName, infos in self.writeInTask.items():
            handler = infos["Handler"]
            indices = infos["indices"]
            handler.set_values_and_save(os.path.join(workPath, fileName), indices, X[indices])
    
    def _extract_series(self, workPath):
        
        cache = {}
        
        for sid, item in self.seriesDict.items():
            item['sim'] = read_extract(workPath, item['simItem'])

            cache[sid] = {'obs': item['obs'], 'sim': item['sim']}
        
        return cache
        
    def _subprocess(self, X, i, batch_id):
        
        workPath = self.runQueue.get()
        
        self._set_values(workPath, X)
        
        record = {}
        record['X'] = X; record['i'] = i; record['batch_id'] = batch_id
        
        try:
            process = subprocess.Popen(
                os.path.join(workPath, self.cfg.basic.exeName),
                cwd = workPath,
                stdin = subprocess.PIPE, 
                stdout = subprocess.PIPE, 
                stderr = subprocess.PIPE,
                text = True)
            process.wait()

            cache = self._extract_series(workPath)
            
            # objectives
            for obj_id in self.cfg.objectives.use:
                obj = self.objectivesDict[obj_id]
                comb = obj['comb']
                metric = obj['metric']
                reduce = obj['reduce']
                vals = {}
                archive = {}             
                val = 0
                for sid, weight in comb.items():
                    m = metric(np.array(cache[sid]['obs']), np.array(cache[sid]['sim']))
                    vals[sid] = m
                    val += m * weight
                    archive[sid] ={'obs': cache[sid]['obs'], 'sim': cache[sid]['sim']}
                    
                if reduce == 'weighted_mean':
                    val /= np.sum(list(comb.values()))
                
                record[obj_id] = {'agg_val': val, 'archive': archive, 'vals': vals}
            
            # diagnostics
            for diag_id in self.cfg.diagnostics.use:
                diag = self.diagnosticsDict[diag_id]
                comb = diag['comb']
                metric = diag['metric']
                reduce = diag['reduce']
                
                vals = {}
                archive = {}
                
                val = 0
                for sid, weight in comb.items():
                    m = metric(np.array(cache[sid]['obs']), np.array(cache[sid]['sim']))
                    vals[sid] = m
                    val += m * weight
                    archive[sid] ={'obs': cache[sid]['obs'], 'sim': cache[sid]['sim']}
                
                val = 0
                for sid, weight in comb.items():
                    val += vals[sid] * weight
                
                if reduce == 'weighted_mean':
                    val /= np.sum(list(comb.values()))
                
                record[diag_id] = {'agg_val': val, 'archive': archive, 'vals': vals}
            record['cache'] = cache
        except Exception as e:
            record['error'] = str(e)
        
        finally:
            self.runQueue.put(workPath)
            if hasattr(self, "reporter"):
                self.reporter.submit(record)
            
        return record