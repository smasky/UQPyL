#Temp
import sys
sys.path.append(".")

import os
import re
import json
import queue
import itertools
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from UQPyL.utility.metrics import r_square
from UQPyL.problems import ProblemABC as Problem

#C++ Module
from .swat_utility import read_value_swat, copy_origin_to_tmp, write_value_to_file, read_simulation

def func_NSE_inverse(true_values, sim_values):
    return -1 * r_square(true_values.reshape(-1,1), sim_values.reshape(-1,1))

def func_RMSE(true_values, sim_values):
    return np.sqrt(np.mean(np.square(true_values-sim_values)))

def func_PCC_inverse(true_values, sim_values):
    return -1 * np.corrcoef(true_values.ravel(), sim_values.ravel())[0,1]

def func_Pbias(true_values, sim_values):
    return np.sum(np.abs(true_values-sim_values)/true_values)*100

def func_KGE_inverse(true_values, sim_values):
    trueValues = true_values.ravel()
    simValues = sim_values.ravel()
    r, _ = pearsonr(trueValues, simValues)
    beta = np.std(simValues) / np.std(trueValues)
    gamma = np.mean(simValues) / np.mean(trueValues)
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return -1 * kge

def func_Mean(true_values, sim_values):
    return np.mean(sim_values)

def func_Sum(true_values, sim_values):
    return np.sum(sim_values)

FUNC = {1: "func_NSE_inverse", 2: "func_RMSE", 3: "func_PCC_inverse", 4: "func_Pbias", 5: "func_KGE_inverse", 6: "func_Mean", 7:"func_Sum"}
VAR = {6: "FLOW", 13: "ORGN", 15: "ORGP", 17: "NO3", 19: "NH4", 21: "NO2",47: "TOT_N", 48: "TOT_P"}
FUNC_TYPE = {1: "NSE", 2:"RMSE", 3:"PCC", 4:"Pbias", 5:"KGE", 6:"Mean", 7:"Sum"}

HRU = ["chm", "gw", "hru", "mgt", "sdr", "sep", "sol", "ops"]
WATERSHED = ["pnd", "rte", "sub", "swq", "wgn", "wus"]

class SWAT_UQ(Problem):
    
    '''
    This class is interface for running SWAT model with UQPyL.
    It can calibrate the parameters of SWAT model with observed data.
    It can also manage water resources system with multiple objectives using SWAT and UQPyL.
    Importantly, it supports parallel running of SWAT model with multiple instances.
    '''
    
    modelInfos = {}
    observeInfos = {}
    nHRU = 0; nRCH = 0; nSUB = 0
    
    def __init__(self, nInput: int, nOutput: int, workPath: str, paraFileName: str, 
                 evalFileName: str, swatExeName: str, specialParaList: list = None, tempPath:str = None,
                 userObjFunc: callable = None,
                 userConFunc: callable = None, nCons = 0,
                 maxThreads: int = 12, numParallel: int = 5, verboseFlag = False,
                 name: str = None):
        
        self.verboseFlag = verboseFlag

        self.name = name if name is not None else "SWAT-UQ"
        
        #create the space for running multiple instance of SWAT
        if tempPath is None:
            #if dont set the tempPath, create a temp dir in the current working directory
            workDir = os.path.join(os.getcwd(), "temp")
            tempPath = os.path.join(workDir, nowTime)
        nowTime = datetime.now().strftime("%m%d_%H%M%S")
        tempPath = os.path.join(tempPath, nowTime)
        os.makedirs(tempPath)
        self.workTempDir = tempPath

        #basic setting
        self.workPath = workPath
        self.paraFileName = paraFileName
        self.evalFileName = evalFileName
        self.specialParaList = specialParaList
        self.swatExeName = swatExeName
        
        self.maxWorkers = maxThreads
        self.numParallel = numParallel

        self.userObjFunc = userObjFunc
        self.userConFunc = userConFunc
        
        if self.verboseFlag:
            print("="*25 + "basic setting" + "="*25)
            print("The path of SWAT project is: ", self.workPath)
            print("The file name of optimizing parameters is: ", self.paraFileName)
            print("The file name of evaluation(observed) data is: ", self.evalFileName)
            print("The name of SWAT executable is: ", self.swatExeName)
            print("Temporary directory has been created in: ", self.workTempDir)
            print("=" * 70)
            print("\n" * 2)
        
        self._initial()
        self._record_default_values()
        self._get_evalData()
        
        self.workPathQueue = queue.Queue()
        self.workTempDirs = []
        
        for i in range(numParallel):
            path = os.path.join(self.workTempDir, "instance{}".format(i))
            self.workTempDirs.append(path)
            self.workPathQueue.put(path)
                
        with ThreadPoolExecutor(max_workers = self.numParallel) as executor:
            futures = [executor.submit(copy_origin_to_tmp, self.workPath, workTemp) for workTemp in self.workTempDirs]
        
        for future in futures:
            future.result()
        
        if self.nInput != len(self.varName):
            raise ValueError("The number of input variables is not equal to the number of parameters!")
        
        super().__init__(nInput = nInput, nOutput = nOutput, 
                            lb = self.lb, ub = self.ub, varType = self.varType, varSet = self.varSet)

    def evaluate(self, X):
        
        n = X.shape[0]
        nOut = self.nOutput
        objs = np.zeros((n, nOut))
        nCons = self.nConstraints
        cons = np.zeros((n, nCons)) if nCons > 0 else None
        
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
    
    def _subprocess(self, input_x, id):
        
        workPath = self.workPathQueue.get()
        self._set_values(workPath, input_x)
        
        try:
            process = subprocess.Popen(
                os.path.join(workPath, self.swatExeName),
                cwd = workPath,
                stdin = subprocess.PIPE, 
                stdout = subprocess.PIPE, 
                stderr = subprocess.PIPE,
                text = True)
            process.wait()
            
            dataInfos = self.observeInfos["obsData"]
            funcComb = self.observeInfos["funcComb"]
            funcCombTypes = self.observeInfos["funcCombTypes"]
            
            objDict = {}
            consDict = {}
            objSeries = {}
            consSeries = {}
            
            for funcId in funcComb.keys():
                seriesComb = funcComb[funcId]
                funcCombType = funcCombTypes[funcId]
                val = 0
                
                for serID in seriesComb:
                    dataInfo = dataInfos[serID]
                    funcID = dataInfo[1]
                    funcCombType = dataInfo[2]
                    funcType = dataInfo[3]
                    rchId = dataInfo[4]
                    varCol = dataInfo[5]
                    weight = dataInfo[6]
                    readLines = dataInfo[7]
                    observedValue = dataInfo[8]
                    
                    simValueList = []
                    for lines in readLines:
                        startLine = int(lines[0])
                        endLine = lines[1]
                        subValue = np.array(read_simulation(os.path.join(workPath, "output.rch"), varCol+1, rchId, self.modelInfos["nRCH"], startLine, endLine))
                        simValueList.append(subValue)
                        
                    simValue = np.concatenate(simValueList, axis = 0)
                    val += eval(FUNC[funcType])(observedValue, simValue)*weight
                    
                    if funcCombType == "OBJ":
                        objSeries[serID] = (observedValue, simValue)
                    else:
                        consSeries[serID] = (observedValue, simValue)
                
                if funcCombType == "OBJ":
                    objDict[funcID] = val
                else:
                    consDict[funcID] = val  
                
            self.workPathQueue.put(workPath)
            
            #simulation attributes
            attrs = {}
            attrs['id'] = id
            attrs['objs'] = objDict
            attrs['cons'] = consDict
            attrs['objSeries'] = objSeries
            attrs['consSeries'] = consSeries
            attrs['x'] = input_x
            
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
                                         paras_values.ravel())
                futures.append(future)
            
            for future in futures:
                res = future.result()
    
    def _get_evalData(self):
        
        filePath = os.path.join(self.workPath, self.evalFileName)
        
        serIDs = []; rchIDs = []; varCols = []; rchWgts = []; funcTypes = []; data = []; funcCombTypes = {}
        funcCombs = {}
        
        self.obsObjs = 0
        self.obsCons = 0
        
        printFlag = self.modelInfos["printFlag"]
        
        try:
            with open(filePath, "r") as f:
                
                lines = f.readlines()
                
                patternSeries = re.compile(r'SER_(\d+)\s+')
                patternFunc = re.compile(r'([a-zA-Z]*)_(\d+)\s+')
                patternReach = re.compile(r'REACH_ID_(\d+)\s+')
                patternCol = re.compile(r'VAR_COL_(\d+)\s+')
                patternType = re.compile(r'TYPE_(\d+)\s+')
                patternValue = re.compile(r'(\d+)\s+[a-zA-Z]*_?OUT_(\d+)_(\d+)\s+(\d+\.?\d*)')
                
                i = 0
                while i < len(lines):
                    line = lines[i]
                    
                    matchSeries = patternSeries.match(line)
                    
                    if matchSeries:
                    
                        serID = int(matchSeries.group(1))
                        if serID in serIDs:
                            raise ValueError("The series ID is duplicated, please check the observed data file!")
                        else:
                            serIDs.append(serID)
                        
                        matchFunc = patternFunc.match(lines[i+1])
                        funcCombType = matchFunc.group(1)
                        if funcCombType == "OBJ":
                            self.obsObjs += 1
                        elif funcCombType == "CON":
                            self.obsCons += 1
                        else:
                            raise ValueError("The function combination type is not valid, only `OBJ` and `CON` are supported, please check the observed data file!")
                        funcID = int(matchFunc.group(2))
                        funcCombTypes[funcID] = funcCombType
                        if funcID not in funcCombs.keys():
                            funcCombs[funcID] = []
                        funcCombs[funcID].append(serID)
                        
                        funcType = int(patternType.match(lines[i+2]).group(1))
                        funcTypes.append(funcType)
                        
                        rchID = int(patternReach.match(lines[i+3]).group(1))
                        rchIDs.append(rchID)
                        
                        varCol = int(patternCol.match(lines[i+4]).group(1))
                        varCols.append(varCol)
                        
                        weight = float(re.search(r'\d+\.?\d*',lines[i+5]).group())
                        rchWgts.append(weight)
                        
                        numData = int(re.search(r'\d+', lines[i+6]).group())
                        
                        i = i+7
                        
                        line = lines[i]
                        while patternValue.match(line) is None:
                            i += 1
                            line = lines[i]   
                               
                        n = 0
                        while True:
                            line = lines[i]; n += 1
                            matchData = patternValue.match(line)
                            _, time, year = map(int, matchData.groups()[:-1])
                            value = float(matchData.groups()[-1])
                            if printFlag == 0:
                                years = year - self.modelInfos["beginDate"].year
                                if years == 0:
                                    index = time - self.modelInfos["beginDate"].month
                                else:
                                    index = time + 12-self.modelInfos["beginDate"].month + (years-1)*12
                            else:
                                index = (datetime(year, 1, 1)+timedelta(days=time-1)-self.modelInfos["beginRecord"]).days
                            data.append([serID, funcID, funcCombType, funcType, rchID, varCol, weight, int(index), int(year), int(time), value])
                            if n == numData:
                                break
                            else:
                                i += 1              
                    i += 1
        except FileNotFoundError:
            raise FileNotFoundError("The observed data file is not found, please check the file name!")
        
        except Exception as e:
            raise ValueError("There is an error in observed data file, please check!")
        
        # dtype = {'series_id': int, 'rch_id': int, 'var_col': int, 'obj_type': int, 'obj_id': int, 'weight': float, 'index': int, 'year': int, 'time': int, 'value': float}                     
        obsData = pd.DataFrame(data, columns=['series_id', 'func_id', 'func_comb_type', 'func_type', 'rch_id', 'var_col', 'weight','index', 'year', 'time', 'value'])        
        dataInfos = {}
        
        for serID in serIDs:
            data = obsData.query('series_id==@serID')
            funcID = data['func_id'].iloc[0]
            funcCombType = data['func_comb_type'].iloc[0]
            funcType = data['func_type'].iloc[0]
            rchID = data['rch_id'].iloc[0]
            varCol = data['var_col'].iloc[0]
            weight = data['weight'].iloc[0]
            dataVal = data['value'].to_numpy(dtype=float)
            dataIndex = data['index'].to_numpy(dtype=int)
            readLines = self._get_lines_for_output_(dataIndex)
            dataInfos[serID] = (serID, funcID, funcCombType, funcType, rchID, varCol, weight, readLines, dataVal) #TODO
        
        self.observeInfos["numSer"] = len(serIDs)
        self.observeInfos["obsData"] = dataInfos
        self.observeInfos["funcComb"] = funcCombs
        self.observeInfos["funcCombTypes"] = funcCombTypes
        
        if self.verboseFlag:
            print("="*25 + "Observed Information" + "="*25)
            print("The number of observed data series is: ", len(serIDs))
            print("The number of objective functions is: ", self.obsObjs)
            print("The number of constraint functions is: ", self.obsCons)
            seriesIDFormatted = "{:^10}".format("Series_ID")
            funcIDFormatted = "{:^10}".format("Func_ID")
            funcCombTypeFormatted = "{:^20}".format("Func_Comb_Type")
            funcTypeFormatted = "{:^10}".format("Func_Type")
            rchIDFormatted = "{:^10}".format("Reach_ID")
            varColFormatted = "{:^10}".format("Var_Col")
            weightFormatted = "{:^10}".format("Weight")
            dataFormatted = "{:<30}".format("readLines")
            print(seriesIDFormatted + "||" + funcIDFormatted + "||" + funcCombTypeFormatted + "||" + funcTypeFormatted + "||"+rchIDFormatted + "||" + varColFormatted + "||" + weightFormatted + "||" + dataFormatted)
            for _, series in funcCombs.items():
                for id in series:
                    seriesIDFormatted = "{:^10}".format(dataInfos[id][0])
                    funcIDFormatted = "{:^10}".format(dataInfos[id][1])
                    funcCombTypeFormatted = "{:^20}".format(dataInfos[id][2])
                    funcTypeFormatted = "{:^10}".format(FUNC_TYPE[dataInfos[id][3]])
                    rchIDFormatted = "{:^10}".format(dataInfos[id][4])
                    varColFormatted = "{:^10}".format(dataInfos[id][5])
                    weightFormatted = "{:^10}".format(dataInfos[id][6])
                    readLines = dataInfos[id][7]
                    lineStr = ""
                    for line in readLines:
                        lineStr += str(line[0])+"-"+str(line[1])+" "
                    dataFormatted = "{:<30}".format(lineStr)
                    print(seriesIDFormatted+"||"+funcIDFormatted+"||"+funcCombTypeFormatted+"||"+funcTypeFormatted+"||"+rchIDFormatted+"||"+varColFormatted+"||"+weightFormatted+"||"+dataFormatted)
            print("="*70)
        
    def _record_default_values(self):
        """
        record default values from the swat file
        """
        
        varInfosPath = os.path.join(self.workPath, self.paraFileName)
        LB=[]; UB=[]; varType=[]; varSet={}; varName=[]; varMode=[]; setHruID=[]
        
        with open(varInfosPath, 'r') as f:
            
            lines = f.readlines()
            for i, line in enumerate(lines):
                
                tmpList = line.split()
                name = tmpList[0]
                mode = tmpList[1]
                T = tmpList[2]
                LB_UB =  tmpList[3].split("_")
                HRUID = tmpList[4:]
                
                varName.append(name)
                setHruID.append(HRUID)
                
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
        self.xLabels = self.varName
        self.varSet = varSet
        self.varType = varType
        self.nInput = len(self.varName)
        
        if self.verboseFlag:
            print("="*50+"Parameter Information"+"="*50)
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
                HruFormatted = "{:^20}".format(" ".join(setHruID[i]))
                print(nameFormatted+"||"+typeFormatted+"||"+modeFormatted+"||"+LBFormatted+"||"+UBFormatted+"||"+HruFormatted)
            print("="*120)
            print("\n"*1)
            
        self.varInfos = {}
        
        watershedToHru = self.modelInfos["watershedToHru"]
        watershedList = self.modelInfos["watershedList"]
        hruList = self.modelInfos["hruList"]
        
        for i, element in enumerate(self.varName):
            
            suffix = self.parasInfos.query('para_name==@element')['file_name'].values[0]
            position = self.parasInfos.query('para_name==@element')['position'].values[0]
            
            if(self.parasInfos.query('para_name==@element')['type'].values[0] == "int"):
                varType = 0 #integer
            else:
                varType = 1 #float
            
            if suffix in HRU:
                if setHruID[i][0] == "all":
                    files = [e+".{}".format(suffix) for e in hruList]
                else:
                    files = []
                    for comb in setHruID[i]:
                        if "(" not in comb:
                            code = f"{'0' * (9 - 4 - len(comb))}{comb}{'0'*4}"
                            for hru in watershedToHru[code]:
                                files.append(f"{hru}.{suffix}")
                        else:
                            sub = comb.split("(")[0]
                            hru = comb.split("(")[1].strip(")").split(',')
                            for e in hru:
                                code = f"{'0' * (9 - 4 - len(sub))}{sub}{'0'*(4-len(e))}{e}"
                                files.append(f"{code}.{suffix}")
                                
            elif suffix in WATERSHED:
                if setHruID[i][0] == "all":
                    files = [e+"."+suffix for e in watershedList]
                else:
                    files = []
                    for e in setHruID[i]: 
                        code = f"{'0' * (9 - 4 - len(e))}{e}{'0'*4}"
                        files.append(code+"."+suffix)
                    
            elif suffix == "bsn":
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
        
        with ThreadPoolExecutor(max_workers = self.maxWorkers) as executor:
            futures = []
            for fileName, infos in self.varInfos.items():
                futures.append(executor.submit(read_value_swat, self.workPath, fileName, infos["name"], infos["position"], 1))
        
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
        It reads the control file fig.fig and records the model information.
        '''
        paras = ["IPRINT", "NBYR", "IYR", "IDAF", "IDAL", "NYSKIP"]
        pos = ["default"] * len(paras)
        dictValues = read_value_swat(self.workPath, "file.cio", paras, pos, 0)
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
        watershed = {}
        with open(os.path.join(self.workPath, "fig.fig"), "r") as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(r'(\d+)\.sub', line)
                if match:
                    watershed[match.group(1)] = []
        
        #read sub files
        for sub in watershed:
            fileName = sub + ".sub"
            with open(os.path.join(self.workPath, fileName), "r", encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    match = re.search(r'(\d+)\.mgt', line)
                    if match:
                        watershed[sub].append(match.group(1))
        
        self.modelInfos["watershedList"] = list(watershed.keys())
        self.modelInfos["hruList"] = list(itertools.chain.from_iterable(watershed.values()))
        self.modelInfos["watershedToHru"] = watershed
        self.modelInfos["nHRU"] = len(self.modelInfos["hruList"])
        self.modelInfos["nWatershed"] = len(self.modelInfos["watershedList"])
        self.modelInfos["nRCH"] = len(self.modelInfos["watershedList"])
        # self.nRCH = self.modelInfos["nRCH"] #TODO: check if this is correct
        
        #read the paras file
        HEAD = ["para_name",  "type", "file_name", "position"]
        self.parasInfos = pd.DataFrame(self._load_parameters(), columns=HEAD)
        
        #for special paras file
        if self.specialParaList is not None:
            for para in self.specialParaList:
                self.parasInfos.loc[para[0]] = para
        
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
            
    def _get_lines_for_output_(self, index):
        
        index.ravel().sort()
        curGroup = [index[0]]; linesGroup=[]
        
        for i in range(1, len(index)):
            if index[i] == curGroup[-1]+1:
                curGroup.append(index[i])
            else:
                linesGroup += self._generate_data_lines(curGroup)
                curGroup = [index[i]]
        
        linesGroup += self._generate_data_lines(curGroup)
        
        return linesGroup
    
    def _generate_data_lines(self, group):
        
        start = group[0]; end = group[-1]
        printFlag = self.modelInfos["printFlag"]
        nRCH = self.modelInfos["nRCH"]

        lines = []
        if printFlag == 0:
            beginMonth = self.modelInfos["beginRecord"].month
            firstPeriod = 12-beginMonth
            if start <= firstPeriod:
                if end <= firstPeriod:
                    endInYear = end
                    lines.append([10+nRCH*start, 9+nRCH*(endInYear+1)])
                    return lines
                else:
                    endInYear = firstPeriod
                lines.append([10+nRCH*start, 9+nRCH*(endInYear+1)])
            else:
                years= start // 12
                startInYear = start
                endInYear = years*12 + 11
                if end <= endInYear:
                    lines.append([10 + nRCH * startInYear + nRCH * years, 9 + nRCH * (end + 1) + nRCH * years])
                    return lines
                else:
                    lines.append([10 + nRCH * startInYear, 9 + nRCH * (endInYear + 1) + nRCH * years])
            while True:
                startInYear = endInYear + 1
                endInYear = startInYear + 11
                years = (startInYear - firstPeriod) // 12 + 1
                if endInYear >= end:
                    lines.append([10 + nRCH * startInYear + nRCH * years, 9 + nRCH * (end + 1) + nRCH * years])
                    break
                else:
                    lines.append([10 + nRCH * startInYear, 9 + nRCH * (endInYear + 1) + nRCH * years])
            return lines 
        elif printFlag == 1:
            lines = [[10 + nRCH * start, 9 + nRCH * (end + 1)]]
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
#================================================================

