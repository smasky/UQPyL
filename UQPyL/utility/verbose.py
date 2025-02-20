import os
import re
import time
import h5py
import math
import functools
import numpy as np

from prettytable import PrettyTable
            
class Verbose():
    '''
    This is a class for printing and record verbose messages.
    '''
    logLines = None
    logFlag = False
    saveFlag = False
    verbose = False
    workDir = os.getcwd()
    totalWidth = 120
    
    @staticmethod
    def output(text, problem):
        
        if isinstance(text, PrettyTable):
            text = str(text)+'\n'
        
        if problem.logLines is not None:
            problem.logLines.append(text+'\n')
        
        if hasattr(problem, "verboseEmit"):
            if problem.verboseEmit:
                problem.verboseEmit.send(text)
        
        if problem.verboseFlag:
            print(text)
    
    @staticmethod
    def formatTime(seconds): 
        
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600) 
        minutes, seconds = divmod(seconds, 60)
        
        return f"{days} day | {hours} hour | {minutes} minute | {seconds: .2f} second"
    
    @staticmethod
    def verboseMultiSolutions(dec, metric, feasible, FEs, Iters, width, problem):
        
        nDecs = dec.shape[0]
        
        heads = ["FEs", "Iters","OptType", "HV", "Feasible", "Num of Non-dominated Solutions"]
        values = [FEs, Iters, problem.optType]+[ format(metric, ".4f")]+[feasible]+[nDecs]
        
        table = PrettyTable(heads)
        table.add_row([" "]*len(heads))
        headerString = table.get_string(fields=heads, header=True, border=False)
        maxWidth = max(len(line) for line in headerString.splitlines())*1.5
        
        count = math.ceil(maxWidth/width)
        
        tables = Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table, problem)
    
    @staticmethod
    def verboseSingleSolutions(dec, obj, feasible, xLabels, yLabels, FEs, Iters, width, problem):
        
        heads = ["FEs"]+["Iters"]+["OptType"]+["Feasible"]+yLabels+xLabels
        
        values = [FEs, Iters]+[problem.optType]+[feasible]+[format(item, ".1e") for item in obj.ravel()]+[format(item, ".3f") for item in dec.ravel()]
        
        table = PrettyTable(heads)
        table.add_row([" "]*len(heads))
        headerString = table.get_string(fields=heads, header=True, border=False)
        maxWidth = max(len(line) for line in headerString.splitlines())*1.8
        
        count = math.ceil(maxWidth/width)+1
        
        tables = Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table, problem)
    
    @staticmethod
    def verboseTable(heads, values, num, width):
        
        col = math.ceil(len(heads)/num)
        rows = num
        tables = []
        
        for i in range(rows):
            
            if i+1 != rows:
                end = (i+1)*col
            else:
                end = len(heads)

            table = PrettyTable(heads[i*col:end])
            
            table.max_width = int(width/(col+4))
            table.min_width = int(width/(col+4))
            table.add_row(values[i*col:end])
            
            tables.append(table)
            
        return tables
    
    @staticmethod
    def verboseSi(problem, xLabels, Si, width):
        
        heads = xLabels
        values = [format(item, ".4f") for item in Si.ravel()]
        
        table = PrettyTable(heads)
        table.add_row([" "]*len(heads))
        headerString = table.get_string(fields=heads, header=True, border=False)
        maxWidth = max(len(line) for line in headerString.splitlines())*1.5
        
        count = math.ceil(maxWidth/width)
        
        tables = Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table, problem)
                
    @staticmethod
    def decoratorRecord(func):
        
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            
            problem = obj.problem
            if hasattr(problem, 'GUI'):
                totalWidth = problem.totalWidth
            else:
                totalWidth = Verbose.totalWidth
                
            func(obj, *args, **kwargs) # Main Process
            
            if obj.verboseFlag and obj.iters%obj.verboseFreq==0:
                title = "FEs: "+str(obj.FEs)+" | Iters: "+str(obj.iters)
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                if obj.problem.nOutput == 1:
                    Verbose.verboseSingleSolutions(obj.result.bestTrueDecs, obj.result.bestTrueObjs, obj.result.bestFeasible, obj.problem.xLabels, obj.problem.yLabels, obj.FEs, obj.iters, totalWidth, problem)
                else:
                    Verbose.verboseMultiSolutions(obj.result.bestTrueDecs, obj.result.bestMetric, obj.result.bestFeasible, obj.FEs, obj.iters, totalWidth, problem)
        
        return wrapper
    
    @staticmethod
    def saveData(obj, folderData, type=1):
        
        if type == 0:
            filename = f"{obj.name}_{obj.problem.name}"
        else:
            filename = f"{obj.name}_{obj.problem.name}_D{obj.problem.nInput}_M{obj.problem.nOutput}"

        allFiles = [f for f in os.listdir(folderData) if os.path.isfile(os.path.join(folderData, f))]
        
        pattern = f"{filename}_(\d+)"
        
        maxNum = 0
        for file in allFiles:
            match = re.match(pattern, file)
            if match:
                number = int(match.group(1))
                if number > maxNum:
                    maxNum = number
        maxNum += 1
        
        filename += f"_{maxNum}.hdf"
        
        filepath = os.path.join(folderData, filename)
        
        resultHDF5 = obj.result.generateHDF5()
        
        text = f"Result Save Path: {filepath}"
        
        if obj.problem.logLines is not None:
            obj.problem.logLines.append(text)
        
        if hasattr(obj.problem, 'GUI'):
            obj.problem.verboseEmit.send(text)
        
        with h5py.File(filepath, 'w') as f:
            save_dict_to_hdf5(f, resultHDF5)
    
    @staticmethod
    def saveLog(obj, folderLog, type = 1):
        
        if type == 0:
            filename= f"{obj.name}_{obj.problem.name}"
        else:
            filename = f"{obj.name}_{obj.problem.name}_D{obj.problem.nInput}_M{obj.problem.nOutput}"

        allFiles = [f for f in os.listdir(folderLog) if os.path.isfile(os.path.join(folderLog, f))]
        
        pattern = f"{filename}_(\d+)"
        
        maxNum = 0
        for file in allFiles:
            match = re.match(pattern, file)
            if match:
                number = int(match.group(1))
                if number > maxNum:
                    maxNum = number
        maxNum += 1
        
        filename += f"_{maxNum}.txt"
        
        filepath = os.path.join(folderLog, filename)
        
        with open(filepath, "w") as f:
            f.writelines(obj.problem.logLines)
    
    @staticmethod
    def decoratorRun(func):
                
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            
            if len(args) > 0:
                problem = args[0]
            elif 'problem' in kwargs:
                problem = kwargs['problem']
            problem.verboseFlag = obj.verboseFlag
            totalWidth = Verbose.totalWidth
            
            if obj.logFlag or hasattr(problem, 'GUI'):
                problem.logLines = []
            else:  
                problem.logLines = None
            
            if obj.verboseFlag or obj.logFlag:
                if hasattr(problem, 'GUI'):
                    totalWidth = problem.totalWidth
                else:
                    try:
                        totalWidth = os.get_terminal_size().columns
                        Verbose.totalWidth = totalWidth
                    except Exception:
                        Verbose.totalWidth = totalWidth
            
            if obj.logFlag or obj.saveFlag:
                
                if hasattr(problem, 'GUI'):
                    workDir = problem.workDir
                    folderData, folderLog = Verbose.checkDir(workDir) 
                else:
                    folderData, folderLog = Verbose.checkDir(Verbose.workDir)
                
            #TODO            
            if  obj.verboseFlag or problem.logLines:
                
                title = obj.name+" Setting"
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                
                keys = obj.setting.keys
                values = obj.setting.values
                table = PrettyTable(keys)
                table.add_row(values)
                Verbose.output(table, problem)
                
            if hasattr(problem, 'GUI'):
                iterEmit = problem.iterEmit
                iterEmit.send()
            
            startTime = time.time()
            res = func(obj, *args, **kwargs)
            endTime = time.time()
            totalTime = endTime-startTime
            
            if obj.verboseFlag:
                
                title = "Conclusion"
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                Verbose.output("Time:  "+Verbose.formatTime(totalTime), problem)
                Verbose.output(f"Used FEs:    {obj.FEs}  |  Iters:  {obj.iters}", problem)
                Verbose.output(f"Best Objs and Best Decision with the FEs", problem)
                
                if obj.problem.nOutput == 1:
                    Verbose.verboseSingleSolutions(res.bestTrueDecs, res.bestTrueObjs, res.bestFeasible, obj.problem.xLabels, obj.problem.yLabels, res.appearFEs, res.appearIters, totalWidth, problem)
                else:
                    Verbose.verboseMultiSolutions(res.bestTrueDecs, res.bestMetric, res.bestFeasible, res.appearFEs, res.appearIters, totalWidth, problem)
                    
            if obj.saveFlag:
                
                Verbose.saveData(obj, folderData)
                
            if obj.logFlag:
                
                Verbose.saveLog(obj, folderLog)

            #TODO
            if hasattr(problem, 'GUI'):
                if problem.isStop:
                    iterEmit.unfinished()
                else:
                    iterEmit.finished()
            return res
        return wrapper 
    
    @staticmethod
    def checkDir(workDir):
        
        folder = os.path.join(workDir, "Result")
        
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        folderData = os.path.join(folder, "Data")
        folderLog = os.path.join(folder, "Log")
        
        if not os.path.exists(folderData):
            os.mkdir(folderData)
            
        if not os.path.exists(folderLog):
            os.mkdir(folderLog)
        
        return folderData, folderLog

    @staticmethod
    def decoratorAnalyze(func):
        
        def wrapper(obj, *args, **kwargs):
            
            if len(args) > 0:
                problem = args[0]
            elif 'problem' in kwargs:
                problem = kwargs['problem']
            problem.verboseFlag = obj.verboseFlag
            
            totalWidth = Verbose.totalWidth
            
            if obj.logFlag or hasattr(problem, 'GUI'):
                
                problem.logLines = []
            
            else:
                
                problem.logLines = None
            
            if obj.logFlag or obj.saveFlag:
                
                if hasattr(problem, 'GUI'):
                    totalWidth = problem.totalWidth
                    workDir = problem.workDir
                    folderData, folderLog = Verbose.checkDir(workDir) 
                else:
                    folderData, folderLog = Verbose.checkDir(Verbose.workDir)
            
            if obj.verboseFlag or obj.logFlag:
                
                title = obj.name+" Setting"
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)

                keys = obj.setting.keys()
                values = obj.setting.values()
                
                table = PrettyTable(keys)
                table.add_row(values)
                Verbose.output(table, problem)
                
                title = "Attribute"
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                
                Verbose.output(f"First Order Sensitivity: {obj.firstOrder}", problem)
                Verbose.output(f"Second Order Sensitivity: {obj.secondOrder}", problem)
                Verbose.output(f"Total Order Sensitivity: {obj.totalOrder}", problem)
                
            res = func(obj, *args, **kwargs)
            
            if obj.verboseFlag or obj.logFlag:
      
                title = "Conclusion"
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                
                for key, values in obj.result.Si.items():
                    title = key
                    spacing = int((totalWidth-len(title))/2)-1
                    Verbose.output("-"*spacing+title+"-"*spacing, problem)
                    Verbose.verboseSi(problem, values[0], values[1], Verbose.totalWidth)
                    
            if obj.logFlag:
                Verbose.saveLog(obj, folderLog, type=0)
            
            if obj.saveFlag:
                Verbose.saveData(obj, folderData, type=0)
                
            return res
        return wrapper
    
def save_dict_to_hdf5(h5file, d):
    
    for key, value in d.items():
        if isinstance(value, dict):
            group = h5file.create_group(str(key))
            save_dict_to_hdf5(group, value)
        elif isinstance(value, np.ndarray):
            h5file.create_dataset(key, data = value)
        else:
            h5file.create_dataset(key, data = np.array(value))