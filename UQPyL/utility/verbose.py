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
    logLines=None
    logFlag=False
    saveFlag=False
    verbose=False
    workDir=os.getcwd()
    totalWidth = 120
    
    @staticmethod
    def output(text, problem):
        
        if isinstance(text, PrettyTable):
            text=str(text)+'\n'
        
        if problem.logLines is not None:
            problem.logLines.append(text)
        
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
    def verboseMultiSolutions(dec, obj, FEs, Iters, width, problem):
        
        nDecs=dec.shape[0]
        if len(obj)==1:
            y_labels=["HV"]
        else:
            y_labels=["HV", "IGD"]
        
        heads=["FEs"]+["Iters"]+y_labels+["Num_Non-dominated_Solution"]
        values=[FEs, Iters]+[ format(item, ".4f") for item in obj]+[nDecs]
        
        table=PrettyTable(heads)
        table.add_row([" "]*len(heads))
        headerString = table.get_string(fields=heads, header=True, border=False)
        maxWidth = max(len(line) for line in headerString.splitlines())*1.5
        
        count=math.ceil(maxWidth/width)
        
        tables=Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table, problem)
    
    @staticmethod
    def verboseSingleSolutions(dec, obj, x_labels, y_labels, FEs, Iters, width, problem):
        
        heads=["FEs"]+["Iters"]+y_labels+x_labels
        
        values=[FEs, Iters]+[ format(item, ".2e") for item in obj.ravel()]+[format(item, ".4f") for item in dec.ravel()]
        
        table=PrettyTable(heads)
        table.add_row([" "]*len(heads))
        headerString = table.get_string(fields=heads, header=True, border=False)
        maxWidth = max(len(line) for line in headerString.splitlines())*1.5
        
        count=math.ceil(maxWidth/width)
        
        tables=Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table, problem)
    
    @staticmethod
    def verboseTable(heads, values, num, width):
        
        col=len(heads)//num
        rows=num
        tables=[]
        
        for i in range(rows):
            
            if i+1!=rows:
                end=(i+1)*col
            else:
                end=len(heads)

            table=PrettyTable(heads[i*col:end])
            
            table.max_width=int(width/(col+4))
            table.min_width=int(width/(col+4))
            table.add_row(values[i*col:end])
            
            tables.append(table)
            
        return tables
    
    @staticmethod
    def verboseSi(problem, x_labels, Si, width):
        
        heads=x_labels
        values=[format(item, ".4f") for item in Si.ravel()]
        
        table=PrettyTable(heads)
        table.add_row([" "]*len(heads))
        headerString = table.get_string(fields=heads, header=True, border=False)
        maxWidth = max(len(line) for line in headerString.splitlines())*1.5
        
        count=math.ceil(maxWidth/width)
        
        tables=Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table, problem)
                
    @staticmethod
    def decoratorRecord(func):
        
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            
            problem=obj.problem
            if hasattr(problem, 'GUI'):
                totalWidth=problem.totalWidth
            else:
                totalWidth=Verbose.totalWidth
                
            func(obj, *args, **kwargs)
            
            if obj.verbose and obj.iters%obj.verboseFreq==0:
                title="FEs: "+str(obj.FEs)+" | Iters: "+str(obj.iters)
                spacing=int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                if obj.problem.nOutput==1:
                    Verbose.verboseSingleSolutions(obj.result.bestDec, obj.result.bestObj, obj.problem.x_labels, obj.problem.y_labels, obj.FEs, obj.iters, totalWidth, problem)
                else:
                    Verbose.verboseMultiSolutions(obj.result.bestDec, obj.result.bestMetric, obj.FEs, obj.iters, totalWidth, problem)
        return wrapper
    
    @staticmethod
    def saveData(obj, folder_data, type=1):
        
        if type==0:
            filename= f"{obj.name}_{obj.problem.name}"
        else:
            filename = f"{obj.name}_{obj.problem.name}_D{obj.problem.nInput}_M{obj.problem.nOutput}"

        all_files = [f for f in os.listdir(folder_data) if os.path.isfile(os.path.join(folder_data, f))]
        
        pattern = f"{filename}_(\d+)"
        
        max_num=0
        for file in all_files:
            match = re.match(pattern, file)
            if match:
                number = int(match.group(1))
                if number > max_num:
                    max_num=number
        max_num+=1
        
        filename+=f"_{max_num}.hdf"
        
        filepath = os.path.join(folder_data, filename)
        
        resultHDF5=obj.result.generateHDF5()
        
        text=f"Result Save Path: {filepath}"
        
        if obj.problem.logLines is not None:
            obj.problem.logLines.append(text)
        
        if hasattr(obj.problem, 'GUI'):
            obj.problem.verboseEmit.send(text)
        
        with h5py.File(filepath, 'w') as f:
            save_dict_to_hdf5(f, resultHDF5)
    
    @staticmethod
    def saveLog(obj, folder_log, type=1):
        
        if type==0:
            filename= f"{obj.name}_{obj.problem.name}"
        else:
            filename = f"{obj.name}_{obj.problem.name}_D{obj.problem.nInput}_M{obj.problem.nOutput}"

        all_files = [f for f in os.listdir(folder_log) if os.path.isfile(os.path.join(folder_log, f))]
        
        pattern = f"{filename}_(\d+)"
        
        max_num=0
        for file in all_files:
            match = re.match(pattern, file)
            if match:
                number = int(match.group(1))
                if number > max_num:
                    max_num=number
        max_num+=1
        
        filename+=f"_{max_num}.txt"
        
        filepath = os.path.join(folder_log, filename)
        
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
            problem.verboseFlag=obj.verbose
            totalWidth=Verbose.totalWidth
            
            if obj.logFlag or hasattr(problem, 'GUI'):
                problem.logLines=[]
            else:  
                problem.logLines=None
            
            if obj.verbose or obj.logFlag:
                if hasattr(problem, 'GUI'):
                    totalWidth=problem.totalWidth
                else:
                    totalWidth=os.get_terminal_size().columns
                    Verbose.totalWidth=totalWidth
            
            if obj.logFlag or obj.saveFlag:
                
                if hasattr(problem, 'GUI'):
                    workDir=problem.workDir
                    folder_data, folder_log=Verbose.checkDir(workDir) 
                else:
                    folder_data, folder_log=Verbose.checkDir(Verbose.workDir)
                
            #TODO            
            if  obj.verbose or problem.logLines:
                
                title=obj.name+" Setting"
                spacing=int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                
                keys=obj.setting.keys
                values=obj.setting.values
                table=PrettyTable(keys)
                table.add_row(values)
                Verbose.output(table, problem)
                
            if hasattr(problem, 'GUI'):
                iterEmit=problem.iterEmit
                iterEmit.send()
            
            startTime=time.time()
            res=func(obj, *args, **kwargs)
            endTime=time.time()
            totalTime=endTime-startTime
            
            if obj.verbose:
                
                title="Conclusion"
                spacing=int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                Verbose.output("Time:  "+Verbose.formatTime(totalTime), problem)
                Verbose.output(f"Used FEs:    {obj.FEs}  |  Iters:  {obj.iters}", problem)
                Verbose.output(f"Best Objs and Best Decision with the FEs", problem)
                
                if obj.problem.nOutput==1:
                    Verbose.verboseSingleSolutions(res.bestDec, res.bestObj, obj.problem.x_labels, obj.problem.y_labels, res.appearFEs, res.appearIters, totalWidth, problem)
                else:
                    Verbose.verboseMultiSolutions(res.bestDec, res.bestMetric, res.appearFEs, res.appearIters, totalWidth, problem)

            if obj.saveFlag:
                
                Verbose.saveData(obj, folder_data)
                
            if obj.logFlag:
                
                Verbose.saveLog(obj, folder_log)

            #TODO
            if hasattr(problem, 'GUI'):
                if problem.isStop:
                    iterEmit.unfinished()
                else:
                    iterEmit.finished()
                    
            # if Verbose.isStop!=None:
            #     if Verbose.isStop:
                    
            #         Verbose.iterEmit.unfinished()
                    
            #     else:
                    
            #         Verbose.iterEmit.finished()
                    
            # Verbose.logFlag, Verbose.verbose, Verbose.saveFlag=record 
            return res
        return wrapper 
    
    @staticmethod
    def checkDir(workDir):
        
        folder=os.path.join(workDir, "Result")
        
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        folder_data=os.path.join(folder, "Data")
        folder_log=os.path.join(folder, "Log")
        
        if not os.path.exists(folder_data):
            os.mkdir(folder_data)
            
        if not os.path.exists(folder_log):
            os.mkdir(folder_log)
        
        return folder_data, folder_log

    @staticmethod
    def decoratorAnalyze(func):
        
        def wrapper(obj, *args, **kwargs):
            
            if len(args) > 0:
                problem = args[0]
            elif 'problem' in kwargs:
                problem = kwargs['problem']
            problem.verboseFlag=obj.verbose
            
            totalWidth=Verbose.total_width
            
            if obj.logFlag or hasattr(problem, 'GUI'):
                
                problem.logLines=[]
            
            else:
                
                problem.logLines=None
            
            if obj.logFlag or obj.saveFlag:
                
                if hasattr(problem, 'GUI'):
                    totalWidth=problem.totalWidth
                    workDir=problem.workDir
                    folder_data, folder_log=Verbose.checkDir(workDir) 
                else:
                    folder_data, folder_log=Verbose.checkDir(Verbose.workDir)
            
            if obj.verbose or obj.logFlag:
                
                title=obj.name+" Setting"
                spacing=int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)

                keys=obj.setting.keys()
                values=obj.setting.values()
                
                table=PrettyTable(keys)
                table.add_row(values)
                Verbose.output(table, problem)
                
                title="Attribute"
                spacing=int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                
                Verbose.output(f"First Order Sensitivity: {obj.firstOrder}", problem)
                Verbose.output(f"Second Order Sensitivity: {obj.secondOrder}", problem)
                Verbose.output(f"Total Order Sensitivity: {obj.totalOrder}", problem)
                
            res=func(obj, *args, **kwargs)
            
            if obj.verbose or obj.logFlag:
      
                title="Conclusion"
                spacing=int((totalWidth-len(title))/2)-1
                Verbose.output("="*spacing+title+"="*spacing, problem)
                
                for key, values in obj.result.Si.items():
                    title=key
                    spacing=int((totalWidth-len(title))/2)-1
                    Verbose.output("-"*spacing+title+"-"*spacing, problem)
                    Verbose.verboseSi(problem, values[0], values[1], Verbose.total_width)
                    
            if obj.logFlag:
                
                Verbose.saveLog(obj, folder_log, type=0)
            
            if obj.saveFlag:
                
                Verbose.saveData(obj, folder_data, type=0)
            
            return res
        return wrapper
    
def save_dict_to_hdf5(h5file, d):
    for key, value in d.items():
        if isinstance(value, dict):
            group = h5file.create_group(str(key))
            save_dict_to_hdf5(group, value)  
        elif isinstance(value, np.ndarray):
            h5file.create_dataset(key, data=value)
        else:
            h5file.create_dataset(key, data=np.array(value))