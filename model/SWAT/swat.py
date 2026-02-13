import os
import re
import json
import queue
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from UQPyL.utility.metrics import r_square
from UQPyL.problems import ProblemABC as Problem

from .utility import read_value_swat, copy_origin_to_tmp, write_value_to_file, read_simulation

def func_NSE(trueValues, simValues):
    return r_square(trueValues.reshape(-1,1), simValues.reshape(-1,1))

def func_RMSE(trueValues, simValues):
    return np.sqrt(np.mean(np.square(trueValues-simValues)))

def func_PCC(trueValues, simValues):
    return np.corrcoef(trueValues.ravel(), simValues.ravel())[0,1]

def func_Pbias(trueValues, simValues):
    return np.sum(np.abs(trueValues-simValues)/trueValues)*100

def func_KGE(trueValues, simValues):
    trueValues = trueValues.ravel()
    simValues = simValues.ravel()
    r, _ = pearsonr(trueValues, simValues)
    beta = np.std(simValues) / np.std(trueValues)
    gamma = np.mean(simValues) / np.mean(trueValues)
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return kge

def func_Mean(trueValues, simValues):
    return np.mean(simValues, axis = 0)

def func_Sum(trueValues, simValues):
    return np.sum(simValues, axis = 0)

def func_Max(trueValues, simValues):
    return np.max(simValues, axis = 0)

def func_Min(trueValues, simValues):
    return np.min(simValues, axis = 0)

def func_Extract(trueValues, simValues):
    return -1

FUNC = {1: "func_NSE", 2: "func_RMSE", 3: "func_PCC", 4: "func_Pbias", 5: "func_KGE", 6: "func_Mean", 7:"func_Sum", 8:"func_Max", 9:"func_Min", 10: "func_Extract"}
FUNC_TYPE = {1: "NSE", 2:"RMSE", 3:"PCC", 4:"Pbias", 5:"KGE", 6:"Mean", 7:"Sum", 8:"Max", 9:"Min", 10: "None"}

RCH_VAR = {1 : "FLOW_IN", 2 : "FLOW_OUT",  3: "EVAP", 4: "TLOSS", 5: "SED_IN", 6: "SED_OUT"}
HRU_VAR = {6 : "ET", 9 : "PERC"}
SUB_VAR = {4 : "ET", 6 : "PERC"}

HRU_EXT = ["chm", "gw", "hru", "mgt", "sdr", "sep", "sol", "ops"]
SUB_EXT = ["pnd", "rte", "sub", "swq", "wgn", "wus"]

SPECIAL_ALIAS = {"SOL_AWC" :  "Ave", "SOL_K" : "Ksat", "SOL_BD" : "Bulk", "SOL_ALB" : "Albedo", "SOL_E" : "Erosion"}

class SWAT_UQ(Problem):
    '''
    This class is designed for combining SWAT model with UQPyL.
    It can calibrate the parameters of SWAT model with observed data.
    It can also manage water resources system with multiple objectives using SWAT and UQPyL.
    Importantly, it supports parallel running of SWAT model with multiple instances.
    '''
    
    def __init__(self, projectPath: str, swatExeName: str,
                 workPath: str, paraFileName: str, evalFileName: str, specialFileName: list = None,
                 userObjFunc: callable = None, userConFunc: callable = None,
                 maxThreads: int = 12, numParallel: int = 5, nInput = None, nOutput = None, nConstraints = None,
                 verboseFlag = False, name: str = None, optType = 'min'):
        
        self.modelInfos = {}
        
        self.verboseFlag = verboseFlag

        self.name = name if name is not None else "SWAT-UQ"
        
        #create the space for running multiple instance of SWAT
        tempPath = os.path.join(workPath, "tempForParallel")
        if not os.path.isdir(tempPath):
            os.makedirs(tempPath)
        
        nowTime = datetime.now().strftime("%m%d_%H%M%S")
        tempPath = os.path.join(tempPath, nowTime)
        os.makedirs(tempPath)
        self.workTempDir = tempPath
        
        #copy the original project to the temp directory
        self.workOriginPath = os.path.join(tempPath, "origin")
        copy_origin_to_tmp(projectPath, self.workOriginPath)
        self.workValidationPath = None
        
        #basic setting
        self.workPath = workPath
        self.paraFileName = paraFileName
        self.evalFileName = evalFileName
        self.specialFileName = specialFileName
        
        self.projectPath = projectPath
        self.swatExe = swatExeName
        
        self.maxWorkers = maxThreads
        self.numParallel = numParallel

        self.userObjFunc = userObjFunc
        self.userConFunc = userConFunc
        
        if self.verboseFlag:
            print("="*25 + "basic setting" + "="*25)
            print("The path of SWAT project is: ", self.workPath)
            print("The file name of optimizing parameters is: ", self.paraFileName)
            print("The file name of evaluation(observed) data is: ", self.evalFileName)
            print("The name of SWAT executable is: ", self.swatExe)
            print("Temporary directory has been created in: ", self.workTempDir)
            print("=" * 70)
            print("\n" * 1)
        
        self._initial()
        self._record_default_values()
        self._init_eval()
        
        self.workPathQueue = queue.Queue()
        self.workTempDirs = []
        
        for i in range(numParallel):
            path = os.path.join(self.workTempDir, "parallel{}".format(i))
            self.workTempDirs.append(path)
            self.workPathQueue.put(path)
                
        with ThreadPoolExecutor(max_workers = self.numParallel) as executor:
            futures = [executor.submit(copy_origin_to_tmp, self.workOriginPath, workTemp) for workTemp in self.workTempDirs]
        
        for future in futures:
            future.result()
        
        if self.nInput != len(self.varName):
            raise ValueError("The number of input variables is not equal to the number of parameters!")
        
        if nOutput is not None:
            self.nOutput = nOutput
            
        if nConstraints is not None:
            self.nConstraints = nConstraints

            
        
        super().__init__(nInput = self.nInput, nOutput = self.nOutput, nConstraints = self.nConstraints,
                            lb = self.lb, ub = self.ub, xLabels = self.xLabels,
                            varType = self.varType, varSet = self.varSet, optType = optType)

    def evaluate(self, X):
        
        n = X.shape[0]
        nOut = self.nOutput
        objs = np.zeros((n, nOut))
        nCons = self.nConstraints
        cons = np.zeros((n, nCons)) if nCons > 0 else None
        
        # for i in range(n):
        #     attrs = self._subprocess(X[i, :], i)
        #     id = attrs['id']
        #     if self.userObjFunc is None:
        #         objs[id] = np.array(list(attrs['objs'].values()))
        #     else:
        #         objs[id] = self.userObjFunc(attrs)
                
        with ThreadPoolExecutor(max_workers = self.numParallel) as executor:
            futures = [executor.submit(self._subprocess, X[i, :], i) for i in range(n)]
        
            for _ , future in enumerate(futures):
                attrs = future.result()
                
                id = attrs['id']
                
                if self.userObjFunc is None:
                    #use default
                    objs[id] = np.array(list(attrs['objs'].values()))
                else:
                    #use user define
                    objs[id] = self.userObjFunc(attrs)
                
                if nCons > 0:
                    if self.userConFunc is None:
                        cons[id] = np.array(list(attrs['cons'].values()))
                    else:
                        cons[id] = self.userConFunc(attrs)

        return {'objs': objs, 'cons': cons}
    
    def objFunc(self, X):
        n = X.shape[0]
        objs = np.zeros((n, self.nOutput))
        
        with ThreadPoolExecutor(max_workers = self.numParallel) as executor:
            futures = [executor.submit(self._subprocess, X[i, :], i) for i in range(n)]
        
            for _ , future in enumerate(futures):
                attrs = future.result()
                
                id = attrs['id']
                
                if self.userObjFunc is None:
                    #use default
                    objs[id] = np.array(list(attrs['objs'].values()))
                else:
                    #use user define
                    objs[id] = self.userObjFunc(attrs)
            
        return objs
    
    def conFunc(self, X):
        
        n = X.shape[0]
        
        if self.nConstraints > 0:
            cons = np.zeros((n, self.nConstraints))
        else:
            cons = None
            return cons
        
        with ThreadPoolExecutor(max_workers = self.numParallel) as executor:
            futures = [executor.submit(self._subprocess, X[i, :], i) for i in range(n)]
            
            for _ , future in enumerate(futures):
                attrs = future.result()
                
                id = attrs['id']
                
                if self.userConFunc is None:
                    cons[id] = np.array(list(attrs['cons'].values()))
                else:
                    cons[id] = self.userConFunc(attrs)
                    
        return cons
        
    def apply_parameters(self, X, replace = False):
        
        X = X.ravel()
        if replace:
            self._set_values(self.projectPath, X)
            self._set_values(self.workOriginPath, X)
            print(f"The parameters have been applied to {self.projectPath} and {self.workOriginPath}!")
        else:
            self._set_values(self.workOriginPath, X)
            print(f"The parameters have been applied to {self.workOriginPath}!")
    
    
    def extract_series(self, X, seriesFile = None):
        
        X = X.ravel()
        
        if seriesFile is None:
            seriesFile = self.evalFileName
        
        if self.workValidationPath is None:
            self.workValidationPath = os.path.join(self.workTempDir, "validation")
            copy_origin_to_tmp(self.projectPath, self.workValidationPath)
        
        valInfos = self._read_eval(self.workValidationPath,  seriesFile)
        
        self._set_values(self.workValidationPath, X)
        
        process = subprocess.Popen(
                os.path.join(self.workValidationPath, self.swatExe),
                cwd = self.workValidationPath,
                stdin = subprocess.PIPE, 
                stdout = subprocess.PIPE, 
                stderr = subprocess.PIPE,
                text = True)
        process.wait()
        
        serInfos = valInfos["serInfos"]
        optInfos = valInfos["optInfos"]
        
        attrs = {}
        
        objSeries = {}
        objVal = {}
        conSeries = {}
        conVal = {}
        
        for serID, info in serInfos.items():
            for optID, info in optInfos.items():
                combType = info["combType"]
                comb = info["comb"]
                
                if combType == "OBJ":
                    objSeries[optID] = {}
                    combSeries = objSeries[optID]
                    combVal = objVal
                else:
                    conSeries[optID] = {}
                    combSeries = conSeries[optID]
                    combVal = conVal
                
                val = 0
                
                for serID in comb:
                    dataInfo = serInfos[serID]["data"]
                    funcType = serInfos[serID]["funcType"]
                    loc = serInfos[serID]["loc"]
                    locID = serInfos[serID]["locID"]
                    wgt = serInfos[serID]["wgt"]
                    col = serInfos[serID]["col"]
                    readLines = serInfos[serID]["readLines"]
                    obSeries = dataInfo["value"]

                    if loc == "HRU":
                        filePath = os.path.join(self.workValidationPath, "output.hru")
                        totalItem = self.modelInfos["nHRU"]
                        extraCol = 6
                    elif loc == "SUB":
                        filePath = os.path.join(self.workValidationPath, "output.sub")
                        totalItem = self.modelInfos["nSUB"]
                        extraCol = 4
                    elif loc == "RCH":
                        filePath = os.path.join(self.workValidationPath, "output.rch")
                        totalItem = self.modelInfos["nRCH"]
                        extraCol = 5
                        
                    simList = []
                    
                    for rl in readLines:
                        startLine = rl[0]
                        endLine = rl[1]
                        subSim = np.array(read_simulation(filePath, col + extraCol, locID, totalItem, startLine, endLine))
                        simList.append(subSim)
                    
                    simSeries = np.concatenate(simList, axis = 0)
                    val += eval(FUNC[funcType])(obSeries, simSeries) * wgt
                    
                    combSeries[serID] = {"obs": obSeries, "sim": simSeries}
                        
                combVal[optID] = val
                       
            attrs['objs'] = objVal
            attrs['cons'] = conVal
            attrs['objSeries'] = objSeries
            attrs['consSeries'] = conSeries
            
        return attrs
        
    def validate_parameters(self, X, validate_file = None, objFunc = None, conFunc = None):
        
        attrs = self.extract_series(X, validate_file)
            
        if objFunc is None and self.userObjFunc is None:
            objs = np.array(list(attrs['objs'].values()))
        elif objFunc is not None:
            objs = objFunc(attrs)
        else:
            objs = self.userObjFunc(attrs)
        
        if conFunc is None:
            cons = np.array(list(attrs['cons'].values()))
        elif conFunc is not None:
            cons = conFunc(attrs)
        else:
            cons = self.userConFunc(attrs)
        
        return {'objs': objs, 'cons': cons}
             
    def _subprocess(self, inputX, id):
        
        workPath = self.workPathQueue.get()
        self._set_values(workPath, inputX)
        attrs = {}
        try:
            process = subprocess.Popen(
                os.path.join(workPath, self.swatExe),
                cwd = workPath,
                stdin = subprocess.PIPE, 
                stdout = subprocess.PIPE, 
                stderr = subprocess.PIPE,
                text = True)
            process.wait()
            
            serInfos = self.modelInfos["serInfos"]
            optInfos = self.modelInfos["optInfos"]
            
            objSeries = {}
            objVal = {}
            conSeries = {}
            conVal = {}
            
            for optID, info in optInfos.items():
                combType = info["combType"]
                comb = info["comb"]
                
                if combType == "OBJ":
                    objSeries[optID] = {}
                    combSeries = objSeries[optID]
                    combVal = objVal
                else:
                    conSeries[optID] = {}
                    combSeries = conSeries[optID]
                    combVal = conVal
                
                val = 0
                
                for serID in comb:
                    dataInfo = serInfos[serID]["data"]
                    funcType = serInfos[serID]["funcType"]
                    loc = serInfos[serID]["loc"]
                    locID = serInfos[serID]["locID"]
                    wgt = serInfos[serID]["wgt"]
                    col = serInfos[serID]["col"]
                    readLines = serInfos[serID]["readLines"]
                    obSeries = dataInfo["value"]

                    if loc == "HRU":
                        filePath = os.path.join(workPath, "output.hru")
                        totalItem = self.modelInfos["nHRU"]
                        extraCol = 6
                    elif loc == "SUB":
                        filePath = os.path.join(workPath, "output.sub")
                        totalItem = self.modelInfos["nSUB"]
                        extraCol = 4
                    elif loc == "RCH":
                        filePath = os.path.join(workPath, "output.rch")
                        totalItem = self.modelInfos["nRCH"]
                        extraCol = 5
                        
                    simList = []
                    
                    for rl in readLines:
                        startLine = rl[0]
                        endLine = rl[1]
                        subSim = np.array(read_simulation(filePath, col + extraCol, locID, totalItem, startLine, endLine))
                        simList.append(subSim)
                    
                    simSeries = np.concatenate(simList, axis = 0)
                    val += eval(FUNC[funcType])(obSeries, simSeries) * wgt
                    
                    combSeries[serID] = {"obs": obSeries, "sim": simSeries}
                        
                combVal[optID] = val
                
            self.workPathQueue.put(workPath)
            
            attrs['id'] = id
            attrs['HRUInfos'] = self.modelInfos["HRUInfosTable"]
            attrs['objs'] = objVal
            attrs['cons'] = conVal
            attrs['objSeries'] = objSeries
            attrs['conSeries'] = conSeries
            attrs['x'] = inputX
            
        except Exception as e:
            attrs['error'] = e
        
        return attrs
    
    def _set_values(self, workPath, paras_values):
        
        with ThreadPoolExecutor(max_workers = self.maxWorkers) as executor:
            futures = []
            for fileName, infos in self.varInfos.items():
                future = executor.submit(write_value_to_file, workPath, fileName, 
                                         infos["name"], infos["default"], 
                                         infos["index"], infos["mode"],  infos["position"], infos["type"],
                                         paras_values.ravel(), infos["solLayer"])
                futures.append(future)
            
            for future in futures:
                res = future.result()
    
    def _read_eval(self, fileFolder, fileName):
        
        evalInfos = {}
        
        nOutput = 0; nConstraints = 0
        
        filePath = os.path.join(self.workPath, fileName)
        
        serIDs = []; serInfos = {}; optInfos = {}; locInfos = { "HRU": [], "SUB" : [], "RCH" : [] }
        
        printFlag = self.modelInfos["printFlag"]
        
        with open(filePath, "r") as f:
            
            lines = f.readlines()
            
            patternSeries = re.compile(r'SER_(\d+)')
            patternOpt = re.compile(r'([a-zA-Z]*)_(\d+)')
            patternWgt = re.compile(r'WGT_(\d+\.?\d*)')
            patternLoc = re.compile(r'([a-zA-Z]*)_(\d+)')
            patternCol = re.compile(r'COL_(\d+)')
            patternFunc = re.compile(r'FUNC_(\d+)')
            patternValue = re.compile(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.?\d*)')
            patternDate = re.compile(r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s+to\s+(\d{4}[/-]\d{1,2}[/-]\d{1,2})')
            
            try:
            
                i = 0
                while i < len(lines):
        
                    matchSeries = patternSeries.match(lines[i])
                    
                    if matchSeries:
                        #SER_ID
                        serID = int(matchSeries.group(1))
                        if serID in serIDs:
                            raise ValueError("The series ID is duplicated, please check the observed data file!")
                        else:
                            serIDs.append(serID)
                            serInfos[serID] = {}
                            
                        #OBJ_ID OR CON_ID
                        matchOpt = patternOpt.match(lines[i+1])
                        optCombType = matchOpt.group(1)
                        if optCombType == "OBJ":
                            nOutput += 1
                        elif optCombType == "CON":
                            nConstraints += 1
                        else:
                            raise ValueError("The function combination type is not valid, only `OBJ` and `CON` are supported, please check the observed data file!")
                        
                        optID = int(matchOpt.group(2))
                        optInfos.setdefault(optID, {})
                        serInfos[serID]["optID"] = optID
                        optInfos[optID]["combType"] = optCombType
                        optInfos[optID].setdefault("comb", [])
                        optInfos[optID]["comb"].append(serID)
                        
                        #WGT_NUM
                        matchWgt = patternWgt.match(lines[i+2])
                        wgt = float(matchWgt.group(1))
                        serInfos[serID]['wgt'] = wgt
                        
                        #RCH_ID, SUB_ID, HRU_ID
                        matchLoc = patternLoc.match(lines[i+3])
                        loc = matchLoc.group(1)
                        locID = int(matchLoc.group(2))
                        if loc in ["RCH", "SUB", "HRU"]:
                            serInfos[serID]['loc'] = loc
                            serInfos[serID]['locID'] = locID
                            
                        else:
                            raise ValueError("Only `RCH`, `SUB` and `HRU` are supported, please check the load data location!")
                        
                        #COL_NUM
                        matchCol = patternCol.match(lines[i+4])
                        col = int(matchCol.group(1))
                        locInfos[loc].append(col)
                        
                        serInfos[serID]['col'] = len(locInfos[loc]) # Temp Column Number
                        
                        #FUNC_NUM
                        funcType = int(patternFunc.match(lines[i+5]).group(1))
                        serInfos[serID]['funcType'] = funcType
                                    
                        i = i + 6
                        matchDate = patternDate.match(lines[i])
                        
                        if matchDate:
                            date1 = datetime.strptime(matchDate.group(1), "%Y/%m/%d")
                            date2 = datetime.strptime(matchDate.group(2), "%Y/%m/%d")
                            
                            if printFlag == 0:
                                years = date1.year - self.modelInfos["beginRecord"].year
                                if years == 0:
                                    index = date1.month - self.modelInfos["beginRecord"].month
                                else:
                                    index = date1.month  + 12 - self.modelInfos["beginRecord"].month + (years-1)*12
                            else:
                                index = (date1 - self.modelInfos["beginRecord"]).days
                            
                            YEAR = []
                            INDEX = []
                            VALUE = []
                            current = date1
                            if printFlag == 0:
                                year_diff = date2.year - date1.year
                                month_diff = date2.month - date1.month
                                num = year_diff * 12 + month_diff + 1
                            else:
                                num = (date2 - date1).days + 1
                            for _ in range(num):
                                YEAR.append(current.year)
                                INDEX.append(index)
                                VALUE.append(0)
                                index += 1
                        else:    
                        
                            while patternValue.match(lines[i]) is None:
                                i += 1
                            
                            YEAR = []
                            INDEX = []
                            VALUE = []
                            
                            while True:
                                
                                line = lines[i]
                                matchData = patternValue.match(line)
                                
                                if not matchData:
                                    break
                                
                                _, year, month, day = map(int, matchData.groups()[:-1])
                                
                                value = float(matchData.groups()[-1])
                                
                                date = datetime(year, month, day)
                                
                                if printFlag == 0:
                                    years = year - self.modelInfos["beginRecord"].year
                                    if years == 0:
                                        index = date.month - self.modelInfos["beginRecord"].month
                                    else:
                                        index = date.month  + 12 - self.modelInfos["beginRecord"].month + (years-1)*12
                                else:
                                    index = (date - self.modelInfos["beginRecord"]).days
                                
                                YEAR.append(int(year)); INDEX.append(int(index)); VALUE.append(value)
                                
                                i += 1
                                
                                if i >= len(lines):
                                    break
                                
                        YEAR = np.array(YEAR, dtype=int); INDEX = np.array(INDEX, dtype=int); VALUE = np.array(VALUE, dtype=float)
                        readLines = self._get_lines_for_output_(INDEX, loc)
                        serInfos[serID]['readLines'] = readLines
                        serInfos[serID]['data'] = {'year' : YEAR, 'index' : INDEX, 'value' : VALUE}
                        
                    else:
                        i += 1
            
            except FileNotFoundError:
                raise FileNotFoundError("The observed data file is not found, please check the file name!")
            
            except Exception as e:
                raise ValueError("There is an error in {}, please check!".format(filePath))
            
            self._modify_output_file(locInfos, fileFolder)
            
            evalInfos["serInfos"] = serInfos
            evalInfos["optInfos"] = optInfos
            evalInfos["serIDs"] = serIDs
            evalInfos["nOutput"] = nOutput
            evalInfos["nConstraints"] = nConstraints
            
            if self.verboseFlag:
                print("="*25 + "Observed Information" + "="*25)
                print("The number of observed data series is: ", len(serIDs))
                print("The number of objective functions is: ", nOutput)
                print("The number of constraint functions is: ", nConstraints)
                seriesIDFormatted = "{:^10}".format("Series_ID")
                optIDFormatted = "{:^10}".format("Opt_ID")
                optCombTypeFormatted = "{:^20}".format("Opt_Comb_Type")
                funcTypeFormatted = "{:^10}".format("Func_Type")
                locFormatted = "{:^10}".format("Loc_ID")
                varColFormatted = "{:^10}".format("varCol")
                weightFormatted = "{:^10}".format("Weight")
                lineFormatted = "{:<30}".format("readLines")
                print(seriesIDFormatted + "||" + optIDFormatted + "||" + optCombTypeFormatted + "||" + funcTypeFormatted + "||"+locFormatted + "||" + varColFormatted + "||" + weightFormatted + "||" + lineFormatted)
                
                for optID, info in optInfos.items():
                    combType = info["combType"]
                    comb = info["comb"]
                    for serID in comb:
                        serInfo = serInfos[serID]
                        
                        serIDFormatted = "{:^10}".format(serID)
                        optIDFormatted = "{:^10}".format(optID)
                        optCombTypeFormatted = "{:^20}".format(combType)
                        funcTypeFormatted = "{:^10}".format(serInfo["funcType"])
                        locFormatted = "{:^10}".format("{}_{}".format(serInfo["loc"], serInfo["locID"]))
                        varColFormatted = "{:^10}".format(serInfo["col"])
                        weightFormatted = "{:^10}".format(serInfo["wgt"])
                        
                        readLines = serInfo["readLines"]
                        lineStr = ""
                        for line in readLines:
                            lineStr += str(line[0])+"-"+str(line[1])+" "
                        lineFormatted = "{:<30}".format(lineStr)
                        print(serIDFormatted+"||"+optIDFormatted+"||"+optCombTypeFormatted+"||"+funcTypeFormatted+"||"+locFormatted+"||"+varColFormatted+"||"+weightFormatted+"||"+lineFormatted)
                    
            print("="*70)
            
            return evalInfos
            
    def _init_eval(self):
                
        evalInfos = self._read_eval(self.workOriginPath, self.evalFileName)
        
        self.modelInfos["serInfos"] = evalInfos["serInfos"]
        self.modelInfos["optInfos"] = evalInfos["optInfos"]
        self.modelInfos["serIDs"] = evalInfos["serIDs"]
        self.nOutput  = evalInfos["nOutput"]
        self.nConstraints = evalInfos["nConstraints"]
        
    def _record_default_values(self):
        """
        record default values from the swat file
        """
        
        varInfosPath = os.path.join(self.workPath, self.paraFileName)
        LB=[]; UB=[]; varType=[]; varSet={}; varName=[]; varMode=[]; varScope=[]
        xLabels = []; solLayer = []
        
        with open(varInfosPath, 'r') as f:
            
            lines = f.readlines()
            for i, line in enumerate(lines[1:]):
                
                tmpList = line.split()
                name = tmpList[0]
                
                xLabels.append(name)
                
                # handle the sol file
                match_pos = re.search(r'^(.*?)\((\d+)\)$', name)
                if match_pos:
                    name = match_pos.group(1)
                    layer = int(match_pos.group(2))
                else:
                    layer = 0
                if name in SPECIAL_ALIAS:
                        name = SPECIAL_ALIAS[name]
                ext = self.parasInfos.query('para_name==@name')['file_name'].values[0]    
                if ext == "sol":
                    solLayer.append(layer)
                else:
                    solLayer.append(-1)
                        
                mode = tmpList[1]
                T = tmpList[2]
                LB_UB =  tmpList[3].split("_")
                scope = tmpList[4:]
                
                varName.append(name)
                varScope.append(scope)
                
                if mode in ['v', 'r', 'a']:
                    varMode.append(mode)
                else:
                    raise ValueError(f"The {name} mode is not valid, please check the mode!")
                
                if T not in ['f', 'i', 'd']:
                    raise ValueError(f"The {name} type is not valid, please check the type, only `f`, `i`, `d` are supported!")
                
                if T == "f": #float
                    LB.append(float(LB_UB[0]))
                    UB.append(float(LB_UB[1]))
                    varType.append(0)
                    # varSet.append(None)
                    
                elif T == "i": #integer
                    LB.append(float(LB_UB[0]))
                    UB.append(float(LB_UB[1]))
                    varType.append(1)
                    # varSet.append(None)
                    
                else: #discrete
                    LB.append(0)
                    UB.append(1)
                    varType.append(2)
                    L = []
                    for e in LB_UB:
                        if '.' in e:
                            L.append(float(e))
                        else:
                            L.append(int(e))
                    varSet[i] = L
                
        self.lb = np.array(LB).reshape(1,-1)
        self.ub = np.array(UB).reshape(1,-1)
        self.varMode = varMode
        self.varName = varName
        self.xLabels = xLabels
        self.varSet = varSet
        self.varType = varType
        self.nInput = len(self.varName)
        
        if self.verboseFlag:
            print("="*25+"Parameter Information"+"="*25)
            nameFormatted = "{:^20}".format("Parameter name")
            typeFormatted ="{:^7}".format("Type")
            modeFormatted = "{:^7}".format("Mode")
            LBFormatted = "{:^15}".format("Lower bound")
            UBFormatted = "{:^15}".format("Upper bound")
            HruFormatted = "{:^20}".format("HRU ID or Sub_HRU ID")
            print(nameFormatted+"||"+typeFormatted+"||"+modeFormatted+"||"+LBFormatted+"||"+UBFormatted+"||"+HruFormatted)
            for i in range(len(self.varName)):
                nameFormatted = "{:^20}".format(self.varName[i])
                typeFormatted = "{:^7}".format("Float" if self.varType[i]==0 else "int")
                modeFormatted = "{:^7}".format(self.varMode[i])
                LBFormatted = "{:^15}".format(self.lb[0][i])
                UBFormatted = "{:^15}".format(self.ub[0][i])
                HruFormatted = "{:^20}".format(" ".join(varScope[i]))
                print(nameFormatted+"||"+typeFormatted+"||"+modeFormatted+"||"+LBFormatted+"||"+UBFormatted+"||"+HruFormatted)
            print( "=" * 70 )
            print("\n" * 1)
            
        self.varInfos = {}
        
        SUBToHRU = self.modelInfos["SUBToHRU"]
        SUBList = self.modelInfos["SUBList"]
        HRUList = self.modelInfos["HRUList"]
        
        SUB_IDToFileName = self.modelInfos["SUB_IDToFileName"]
        HRU_IDToFileName = self.modelInfos["HRU_IDToFileName"]
        
        for i, element in enumerate(self.varName):
            
            ext = self.parasInfos.query('para_name==@element')['file_name'].values[0]
            position = self.parasInfos.query('para_name==@element')['position'].values[0]
            
            if(self.parasInfos.query('para_name==@element')['type'].values[0] == "int"):
                varType = 0 #integer
            else:
                varType = 1 #float
            
            if ext in HRU_EXT:
                if varScope[i][0] == "all":
                    files = [HRU_IDToFileName[e]+".{}".format(ext) for e in HRUList]
                else:
                    files = []
                    for comb in varScope[i]:
                        if "(" not in comb:
                            SUB_ID = int(comb)
                            for _, HRU_ID in SUBToHRU[SUB_ID].items():
                                files.append(f"{HRU_IDToFileName[HRU_ID]}.{ext}")
                        else:
                            SUB_ID = comb.split("(")[0]
                            HRU_IDs = comb.split("(")[1].strip(")").split(',')
                            for HRU_Local_ID in HRU_IDs:
                                HRU_ID = SUBToHRU[SUB_ID][int(HRU_Local_ID)]
                                files.append(f"{HRU_IDToFileName[HRU_ID]}.{ext}")         
            elif ext in SUB_EXT:
                if varScope[i][0] == "all":
                    files = [f"{SUB_IDToFileName[SUB_ID]}.{ext}" for SUB_ID in SUBList]
                else:
                    files = []
                    for SUB_ID in varScope[i]:
                        files = [f"{SUB_IDToFileName[int(SUB_ID)]}.{ext}" for SUB_ID in SUBList]
                    
            elif ext == "bsn":
                files = ["basins.bsn"]
            
            for file in files:
                self.varInfos.setdefault(file,{})
                self.varInfos[file].setdefault("index", [])
                self.varInfos[file]["index"].append(i)
                self.varInfos[file].setdefault("mode", [])
                if self.varMode[i] == "v":
                    self.varInfos[file]["mode"].append(0)
                elif self.varMode[i] == "r":
                    self.varInfos[file]["mode"].append(1)
                elif self.varMode[i] == "a":
                    self.varInfos[file]["mode"].append(2)
                
                self.varInfos[file].setdefault("name", [])
                self.varInfos[file]["name"].append(element)
                self.varInfos[file].setdefault("position",[])
                self.varInfos[file]["position"].append(position)
                self.varInfos[file].setdefault("type", [])
                self.varInfos[file]["type"].append(varType)
                self.varInfos[file].setdefault("solLayer", [])
                self.varInfos[file]["solLayer"].append(solLayer[i])
                
        with ThreadPoolExecutor(max_workers = self.maxWorkers) as executor:
            futures = []
            for fileName, infos in self.varInfos.items():
                futures.append(executor.submit(read_value_swat, self.projectPath, fileName, infos["name"], infos["position"], 1))
        
        for future in futures:
            res = future.result()
            for key, items in res.items():
                values = ' '.join(str(value) for value in items)
                paraName, fileName = key.split('|')
                self.varInfos[fileName].setdefault("default", {})
                self.varInfos[fileName]["default"][paraName] = values
          
    def _initial(self):
        '''
        This function is used to initialize the model information.
        It reads the control file `file.cio` and `fig.fig`, and records the model information.
        '''
        paras = ["IPRINT", "NBYR", "IYR", "IDAF", "IDAL", "NYSKIP"]
        pos = ["default"] * len(paras)
        dictValues = read_value_swat(self.projectPath, "file.cio", paras, pos, 0)
        beginDate = datetime(int(dictValues["IYR"][0]), 1, 1) + timedelta(int(dictValues['IDAF'][0]) - 1)
        endDate = datetime(int(dictValues["IYR"][0]) + int(dictValues['NBYR'][0]) - 1, 1, 1) + timedelta(int(dictValues['IDAL'][0]) - 1)
        simulationDays = (endDate-beginDate).days + 1
        outputSkipYears = int(dictValues["NYSKIP"][0])
        outputSkipDays = (datetime(int(dictValues["IYR"][0])+outputSkipYears, 1, 1) + timedelta(int(dictValues['IDAF'][0])-1) - beginDate).days
        beginRecord = beginDate + timedelta(outputSkipDays)
        
        self.modelInfos["printFlag"] = int(dictValues["IPRINT"][0])
        self.modelInfos["beginDate"] = beginDate
        self.modelInfos["endDate"] = endDate
        self.modelInfos["outputSkipYears"] = outputSkipYears
        self.modelInfos["simulationDays"] = simulationDays
        self.modelInfos["beginRecord"] = beginRecord
        
        #read control file fig.fig
        SUBToHRU = {}
        HRUInfos = {}
        SUB_IDToFileName = {}
        HRU_IDToFileName = {}
        
        with open(os.path.join(self.projectPath, "fig.fig"), "r") as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                matchID = re.search(r'Subbasin:\s*(\d+)', lines[i])
                if matchID:
                    matchFileName = re.search(r'(\d+)\.sub', lines[i+1])
                    SUB_IDToFileName[int(matchID.group(1))] = matchFileName.group(1)
                    SUBToHRU[int(matchID.group(1))] = {}
                    i += 2
                else:
                    i += 1
        
        #read sub files
        # HRU information table
        HRUInfosTable = pd.DataFrame(columns=["HRU_ID", "SUB_ID", "HRU_Local_ID", "Slope_Low", "Slope_High", "Luse", "Area"])
        SUBInfosTable = pd.DataFrame(columns=["SUB_ID", "Area"])
        for subID, fileName in SUB_IDToFileName.items():
            fileName = fileName + ".sub"
            tempHRU = []
            with open(os.path.join(self.projectPath, fileName), "r", encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                matchArea = re.search(r'^\s*([\d.]+)', lines[1])
                subArea = float(matchArea.group(1))
                
                SUBInfosTable.loc[subID] = [subID, subArea]
                
                for line in lines:
                    match = re.search(r'(\d+)\.hru', line)
                    if match:
                        tempHRU.append(match.group(1))
            
            for HRUFileName in tempHRU:
                fileName = HRUFileName + ".hru"
                with open(os.path.join(self.projectPath, fileName), "r", encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    content = lines[0]
                    matchID = re.findall(r'HRU:\s*(\d+)', content)
                    matchSlope = re.search(r"Slope:\s*(\d+)-(\d+)", content)
                    matchLUse = re.search(r"Luse:([a-zA-Z]+)", content)
                    matchFr = re.search(r'^\s*([\d.]+)', lines[1])
                    
                    area = subArea * float(matchFr.group(1))
                    slope = [int(matchSlope.group(1)), int(matchSlope.group(2))]
                    luse = matchLUse.group(1)
                    
                    HRU_IDToFileName[int(matchID[0])] = HRUFileName
                    SUBToHRU[subID][int(matchID[1])] = int(matchID[0])
                    HRUInfos[int(matchID[0])] = (subID, int(matchID[1]), slope, luse, area)
                    HRUInfosTable.loc[int(matchID[0])] = [int(matchID[0]), subID, int(matchID[1]), slope[0], slope[1], luse, area]
        
        self.modelInfos["SUB_IDToFileName"] = SUB_IDToFileName
        self.modelInfos["HRU_IDToFileName"] = HRU_IDToFileName
        self.modelInfos["HRUInfosTable"] = HRUInfosTable
        self.modelInfos["SUBInfosTable"] = SUBInfosTable
        self.modelInfos["SUBList"] = list(SUB_IDToFileName.keys())
        self.modelInfos["HRUList"] = list(HRU_IDToFileName.keys())
        self.modelInfos["HRUInfos"] = HRUInfos
        self.modelInfos["SUBToHRU"] = SUBToHRU
        self.modelInfos["nHRU"] = len(self.modelInfos["HRUList"])
        self.modelInfos["nSUB"] = len(self.modelInfos["SUBList"])
        self.modelInfos["nRCH"] = len(self.modelInfos["SUBList"])
        
        #read the paras file
        HEAD = ["para_name",  "type", "file_name", "position"]
        self.parasInfos = pd.DataFrame(self._load_parameters(), columns=HEAD)
        
        #for special paras file
        if self.specialFileName is not None:
            with open(os.path.join(self.workPath, self.specialFileName), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    content = line.strip().split()
                    self.parasInfos.loc[len(self.parasInfos)] = content
        
        if self.verboseFlag:
            print("="*25 + "Model Information" + "="*25)
            print("The time period of simulation is: ", self.modelInfos["beginDate"].strftime("%Y%m%d"), " to ", self.modelInfos["endDate"].strftime("%Y%m%d"))
            print("The number of simulation days is: ", self.modelInfos["simulationDays"])
            print("The number of output skip years is: ", self.modelInfos["outputSkipYears"])
            print("The number of HRUs is: ", self.modelInfos["nHRU"])
            print("The number of Reaches is: ", self.modelInfos["nRCH"])
            if self.modelInfos["printFlag"] == 0:
                print("The print flag of SWAT is: ", "monthly")
            else:
                print("The print flag of SWAT is: ", "daily")
            print("=" * 70)
            print("\n" * 1)
    
    def _modify_output_file(self, locInfos, filePath):
        
        for _, locIDs in locInfos.items():
            if len(locIDs) == 0:
                locIDs.append(1)
                
            locIDs.sort()
            
            if len(locIDs) < 20:
                locIDs.extend([0] * (20 - len(locIDs)))

        lines = None
        with open(os.path.join(filePath, "file.cio"), "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                
                if "RCH" in locInfos.keys():
                    matchRch = re.search(r"Reach output variables:", line)
                    if matchRch:
                        lines[i+1] = " "*3 + "   ".join(map(str, locInfos["RCH"])) + "\n"
                if "SUB" in locInfos.keys():
                    matchSub = re.search(r"Subbasin output variables:", line)
                    if matchSub:
                        lines[i+1] = " "*3 + "   ".join(map(str, locInfos["SUB"])) + "\n"
                if "HRU" in locInfos.keys():
                    matchHru = re.search(r"HRU output variables:", line)
                    if matchHru:
                        lines[i+1] = " "*3 + "   ".join(map(str, locInfos["HRU"])) + "\n"
            
        with open(os.path.join(self.workOriginPath, "file.cio"), "w") as f:
            f.writelines(lines)

    def _get_lines_for_output_(self, index, loc):
        
        index.ravel().sort()
        curGroup = [index[0]]; linesGroup=[]
        
        for i in range(1, len(index)):
            if index[i] == curGroup[-1]+1:
                curGroup.append(index[i])
            else:
                linesGroup += self._generate_data_lines(curGroup, loc)
                curGroup = [index[i]]
        
        linesGroup += self._generate_data_lines(curGroup, loc)
        
        return linesGroup
    
    def _generate_data_lines(self, group, loc):
        
        start = group[0]; end = group[-1]
        printFlag = self.modelInfos["printFlag"]
        if loc == "RCH":
            interval = self.modelInfos["nRCH"]
        elif loc == "SUB":
            interval = self.modelInfos["nSUB"]
        elif loc == "HRU":
            interval = self.modelInfos["nHRU"]

        lines = []
        if printFlag == 0:
            beginMonth = self.modelInfos["beginRecord"].month
            firstPeriod = 12 - beginMonth
            if start <= firstPeriod:
                if end <= firstPeriod:
                    endInYear = end
                    lines.append([10 + interval*start, 9 + interval * (endInYear + 1)])
                    return lines
                else:
                    endInYear = firstPeriod
                lines.append([10 + interval*start, 9 + interval * (endInYear + 1)])
            else:
                years= start // 12
                startInYear = start
                endInYear = years*12 + 11
                if end <= endInYear:
                    lines.append([10 + interval * startInYear + interval * years, 9 + interval * (end + 1) + interval * years])
                    return lines
                else:
                    lines.append([10 + interval * startInYear + interval * years, 9 + interval * (endInYear + 1) + interval * years])
            while True:
                startInYear = endInYear + 1
                endInYear = startInYear + 11
                years = (startInYear - firstPeriod) // 12 + 1
                if endInYear >= end:
                    lines.append([10 + interval * startInYear + interval * years, 9 + interval * (end + 1) + interval * years])
                    break
                else:
                    lines.append([10 + interval * (startInYear + years), 9 + interval * (endInYear + 1) + interval * years])
            return lines 
        elif printFlag == 1:
            lines = [[10 + interval * start, 9 + interval * (end + 1)]]
            return lines
        
    def _load_parameters(self, filePath = "swat_parameters.json"):
       
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(module_dir, filePath)
        
            with open(json_path, 'r', encoding='utf-8') as f:
                params_dict = json.load(f)
            return [(p["name"], p["type"], p["file_name"], p["position"]) for p in params_dict]

        except Exception as e:
            raise e