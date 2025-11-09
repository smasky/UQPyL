import os
import re
import time
import h5py
import math
import xarray as xr
import functools

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
    totalWidth = 110
    
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
        
        heads = ["FEs", "Iters","OptType", "HV", "Feasible", "ND Solutions"]
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
    def verboseSingleSolutions(dec, obj, feasible, xLabels, yLabels, FEs, iters, width, problem):
        
        heads = ["FEs"]+["Iters"]+["OptType"]+["Feasible"]+yLabels+xLabels
        
        values = [FEs, iters]+[problem.optType]+[feasible]+[format(item, ".1e") for item in obj.ravel()]+[format(item, ".3f") for item in dec.ravel()]
        
        table = PrettyTable(heads)
        table.add_row([" "]*len(heads))
        headerString = table.get_string(fields=heads, header=True, border=False)
        maxWidth = max(len(line) for line in headerString.splitlines())*1.8
        
        count = math.ceil(maxWidth/width)+1
        
        tables = Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table, problem)
    
    @staticmethod
    def verboseInference(res, problem):
        
        width = Verbose.totalWidth
        
        if problem.nOutput == 1:
            
            iters = res["iter"]
            dec = res["bestDecs"]
            obj = res["bestObjs"]
            
            heads = ["Iter"] + problem.xLabels + ["Best Objs"]
            
            values = [iters] + [format(item, ".5e") for item in dec.ravel()] + [format(item, ".5e") for item in obj.ravel()]

        else:
            
            iters = res["iter"]
            numPareto = res["numPareto"]
            heads = ["Iter", "Num Pareto"]
            values = [iters, numPareto]
            
        table = PrettyTable(heads)
        table.add_row([" "]*len(heads))
        headerString = table.get_string(fields=heads, header=True, border=False)
        
        maxWidth = max(len(line) for line in headerString.splitlines()) * 2
        
        if maxWidth < width:
            count = 1
        else:
            count = math.ceil(maxWidth / width) + 1
        
        tables = Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table, problem)
        
    @staticmethod
    def verboseTable(heads, item, num, width):
        
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
            
            table.add_row(item[i*col:end])
            
            tables.append(table)
            
        return tables
    
    @staticmethod
    def verboseItem(problem, title, labels, values, width):
        
        tables = []
        items = [format(item, ".4f") for item in values]
        table = PrettyTable(labels)
        table.add_row([" "]*len(labels))
        headerString = table.get_string(fields=labels, header=True, border=False)
        maxWidth = max(len(line) for line in headerString.splitlines())*1.5
        count = math.ceil(maxWidth/width)
        
        spacing = int((Verbose.totalWidth-len(title))/2)-1
        tables += ["-"*spacing+title+"-"*spacing]
        tables += Verbose.verboseTable(labels, items, count, width)
        
        for table in tables:
            Verbose.output(table, problem)

    @staticmethod
    def record(func):
        
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
                    Verbose.verboseSingleSolutions(obj.result.bestDecs_True, obj.result.bestObjs_True, obj.result.bestFeasible, obj.problem.xLabels, obj.problem.yLabels, obj.FEs, obj.iters, totalWidth, problem)
                else:
                    Verbose.verboseMultiSolutions(obj.result.bestDecs_True, obj.result.bestMetric, obj.result.bestFeasible, obj.FEs, obj.iters, totalWidth, problem)
        
        return wrapper
    
    @staticmethod
    def saveData(obj, folderData, result_nc):
        
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
        
        filename += f"_{maxNum}.nc"
        
        filepath = os.path.join(folderData, filename)
                
        text = f"Result Save Path: {filepath}"
        
        if obj.problem.logLines is not None:
            obj.problem.logLines.append(text)
        
        if hasattr(obj.problem, 'GUI'):
            obj.problem.verboseEmit.send(text)
        
        Verbose.saveToNetCDF(filepath, result_nc)
          
    @staticmethod
    def saveToNetCDF(filepath, res):
        
        if isinstance(res, xr.Dataset):
            res.to_netcdf(filepath, mode = "a")
        else:
            for key, ds in res.items():
                ds.to_netcdf(filepath, group = key, mode = "a")

    @staticmethod
    def saveLog(obj, folderLog):
        
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
    
    # decorator for optimization methods
    @staticmethod
    def run(func):
                
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
            
            # main process
            startTime = time.time()
            
            res = func(obj, *args, **kwargs)
            
            endTime = time.time()
            
            totalTime = endTime - startTime
            
            obj.result.runtime = totalTime
            
            if obj.verboseFlag:
                
                title = "Conclusion"
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                Verbose.output("Time:  " + Verbose.formatTime(totalTime), problem)
                Verbose.output(f"Used FEs:    {obj.FEs}  |  Iters:  {obj.iters}", problem)
                Verbose.output(f"Best Objs and Best Decision with the FEs", problem)
                
                if obj.problem.nOutput == 1:
                    Verbose.verboseSingleSolutions(res.bestDecs_True, res.bestObjs_True, res.bestFeasible, obj.problem.xLabels, obj.problem.yLabels, res.appearFEs, res.appearIters, totalWidth, problem)
                else:
                    Verbose.verboseMultiSolutions(res.bestDecs_True, res.bestMetric, res.bestFeasible, res.appearFEs, res.appearIters, totalWidth, problem)
            
            result_nc = obj.result.generateNetCDF()
            
            if obj.saveFlag:
                Verbose.saveData(obj, folderData, result_nc)
                
            if obj.logFlag:
                Verbose.saveLog(obj, folderLog)

            #TODO
            if hasattr(problem, 'GUI'):
                if problem.isStop:
                    iterEmit.unfinished()
                else:
                    iterEmit.finished()
            return result_nc
        return wrapper 
    
    # decorator for analysis methods
    @staticmethod
    def analyze(func):
        
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
                    
            result_nc = func(obj, *args, **kwargs)
            
            if obj.verboseFlag or obj.logFlag:
                
                title = obj.name+" Setting"
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)

                keys = obj.setting.keys()
                values = obj.setting.values()
                
                table = PrettyTable(keys)
                table.add_row(values)
                Verbose.output(table, problem)
                               
            if obj.verboseFlag or obj.logFlag:
      
                title = "Conclusion"
                spacing = int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                
                for target, items in obj.result.res['verbose'].items():
                    title = target
                    spacing = int((totalWidth-len(title))/2)-1
                    Verbose.output("-"*spacing+title+"-"*spacing, problem)
                    
                    for indicator, values in items.items():
                        labels = list(values.keys()); labels.remove('array')
                        Verbose.verboseItem(problem, indicator, labels, values['array'], Verbose.totalWidth)
                    
            if obj.logFlag:
                Verbose.saveLog(obj, folderLog)
            
            if obj.saveFlag:
                Verbose.saveData(obj, folderData, result_nc)
                
            return result_nc
        
        return wrapper

    @staticmethod
    def inference(func):
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
            
            res = func(obj, *args, **kwargs)
            
            folderData, folderLog = Verbose.checkDir(Verbose.workDir)

            Verbose.saveData(obj, folderData, res)
            
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



def save_dict_to_hdf5(h5file, d):
    
    for key, value in d.items():
        if isinstance(value, dict):
            group = h5file.create_group(str(key))
            save_dict_to_hdf5(group, value)
        else:
            h5file.create_dataset(key, data = value)
            

# def save_dict_to_nc(ncfile, d):
    
#     for key, value in d.items():
#         if isinstance(value, dict):
#             group = ncfile.create_group(str(key))
#             save_dict_to_nc(group, value)
#         else:
#             ncfile.create_dataset(key, data = value)