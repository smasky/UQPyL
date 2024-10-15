import os
import re
import time
import h5py
import math
import functools
import numpy as np

from datetime import datetime
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
    IterEmit=None
    VerboseEmit=None
    total_width=os.get_terminal_size().columns
    @staticmethod    
    def output(obj):
        
        if isinstance(obj, PrettyTable):
            obj=str(obj)+'\n'
        
        if Verbose.logFlag:
            Verbose.logLines.append(obj)
        
        if Verbose.verbose:
            print(obj)
    
    @staticmethod
    def verboseSetting(obj):
        
        if obj.verbose or obj.logFlag:
            
            title=obj.name+" Setting"
            spacing=int((Verbose.total_width-len(title))/2)
            Verbose.output("="*spacing+title+"="*spacing)
            keys=obj.setting.keys
            values=obj.setting.values
            table=PrettyTable(keys)
            table.add_row(values)
            Verbose.output(table)    
    
    @staticmethod
    def formatTime(seconds): 
        
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600) 
        minutes, seconds = divmod(seconds, 60)
        
        return f"{days} day | {hours} hour | {minutes} minute | {seconds: .2f} second"
    
    @staticmethod
    def verboseMultiSolutions(dec, obj, FEs, Iters, width):
        
        nDecs=dec.shape[0]
        if len(obj)==1:
            y_labels=["HV"]
        else:
            y_labels=["HV", "IGD"]
        
        heads=["FEs"]+["Iters"]+y_labels+["Num_Non-dominated_Solution"]
        values=[FEs, Iters]+[ format(item, ".4f") for item in obj]+[nDecs]
        
        tables=Verbose.verboseTable(heads, values, 10, width)
        
        for table in tables:
            Verbose.output(table)
    
    @staticmethod
    def verboseSingleSolutions(dec, obj, x_labels, y_labels, FEs, Iters, width):
        
        heads=["FEs"]+["Iters"]+y_labels+x_labels
        
        values=[FEs, Iters]+[ format(item, ".2e") for item in obj.ravel()]+[format(item, ".4f") for item in dec.ravel()]
        
        maxWidth=max(len(s) for s in heads)
        count=math.floor(width/maxWidth)-1
        
        tables=Verbose.verboseTable(heads, values, count, width)
        
        for table in tables:
            Verbose.output(table)
    
    @staticmethod
    def verboseTable(heads, values, num, width):
        
        rows=int(len(heads))//num+1
        cols=num
        tables=[]
        
        for i in range(rows):
            if (i+1)*cols<len(heads):
                end=(i+1)*cols
            else:
                end=len(heads)
                
            table=PrettyTable(heads[i*cols:end])
            table.max_width=int(width/(cols+4))
            table.min_width=int(width/(cols+4))
            table.add_row(values[i*cols:end])
            
            tables.append(table)
            
        return tables
    
    @staticmethod
    def verboseSi(x_labels, Si, width):
        
        heads=x_labels
        values=[format(item, ".4f") for item in Si.ravel()]
        
        tables=Verbose.verboseTable(heads, values, 10, width)
        
        for table in tables:
            Verbose.output(table)
                
    @staticmethod
    def decoratorRecord(func):
        
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            
            func(obj, *args, **kwargs)
            
            if obj.verbose and obj.iters%obj.verboseFreq==0:
                title="FEs: "+str(obj.FEs)+" | Iters: "+str(obj.iters)
                spacing=int((Verbose.total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)
                if obj.problem.nOutput==1:
                    Verbose.verboseSingleSolutions(obj.result.bestDec, obj.result.bestObj, obj.problem.x_labels, obj.problem.y_labels, obj.FEs, obj.iters, Verbose.total_width)
                else:
                    Verbose.verboseMultiSolutions(obj.result.bestDec, obj.result.bestMetric, obj.FEs, obj.iters, Verbose.total_width)
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
            f.writelines(Verbose.logLines)
    
    @staticmethod
    def decoratorRun(func):
                
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            
            record=[Verbose.logFlag, Verbose.verbose, Verbose.saveFlag]
            
            Verbose.logFlag=obj.logFlag
            Verbose.verbose=obj.verbose
            Verbose.saveFlag=obj.saveFlag
            
            #Check result dir
            if Verbose.logFlag or Verbose.saveFlag:
                
                folder_data, folder_log=Verbose.checkDir()
                
            if Verbose.logFlag:
                  
                Verbose.logLines=[]
                
            if  Verbose.verbose or Verbose.logFlag:
                
                title=obj.name+" Setting"
                spacing=int((Verbose.total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)
                
                keys=obj.setting.keys
                values=obj.setting.values
                table=PrettyTable(keys)
                table.add_row(values)
                Verbose.output(table)
            
            startTime=time.time()
            res=func(obj, *args, **kwargs)
            endTime=time.time()
            totalTime=endTime-startTime
            
            if Verbose.verbose:
                
                title="Conclusion"
                spacing=int((Verbose.total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)
                Verbose.output("Time:  "+Verbose.formatTime(totalTime))
                Verbose.output(f"Used FEs:    {obj.FEs}  |  Iters:  {obj.iters}")
                Verbose.output(f"Best Objs and Best Decision with the FEs")
                
                if obj.problem.nOutput==1:
                    Verbose.verboseSingleSolutions(res.bestDec, res.bestObj, obj.problem.x_labels, obj.problem.y_labels, res.appearFEs, res.appearIters, Verbose.total_width)
                else:
                    Verbose.verboseMultiSolutions(res.bestDec, res.bestMetric, res.appearFEs, res.appearIters, Verbose.total_width)

            if Verbose.saveFlag:
                
                Verbose.saveData(obj, folder_data)
                
            if Verbose.logFlag:
                
                Verbose.saveLog(obj, folder_log)

            Verbose.logFlag, Verbose.verbose, Verbose.saveFlag=record
            
            return res
        return wrapper 
    
    @staticmethod
    def checkDir():
        folder=os.path.join(Verbose.workDir, "Result")
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
            
            Verbose.verbose=obj.verbose
            Verbose.logFlag=obj.logFlag
            Verbose.saveFlag=obj.saveFlag
            
            if Verbose.saveFlag:
                Verbose.logLines=[]
            
            if Verbose.logFlag or Verbose.saveFlag:
                
                folder_data, folder_log=Verbose.checkDir()
            
            if Verbose.verbose or Verbose.logFlag:

                title=obj.name+" Setting"
                spacing=int((Verbose.total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)

                keys=obj.setting.keys()
                values=obj.setting.values()
                
                table=PrettyTable(keys)
                table.add_row(values)
                Verbose.output(table)
                
                title="Attribute"
                spacing=int((Verbose.total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)
                
                Verbose.output(f"First Order Sensitivity: {obj.firstOrder}")
                Verbose.output(f"Second Order Sensitivity: {obj.secondOrder}")
                Verbose.output(f"Total Order Sensitivity: {obj.totalOrder}")
                
            res=func(obj, *args, **kwargs)
            
            if Verbose.verbose or Verbose.logFlag:
      
                title="Conclusion"
                spacing=int((Verbose.total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)
                
                for key, values in obj.result.Si.items():
                    title=key
                    spacing=int((Verbose.total_width-len(title))/2)
                    Verbose.output("-"*spacing+title+"-"*spacing)
                    Verbose.verboseSi(values[0], values[1], Verbose.total_width)
                    
            if Verbose.logFlag:
                
                Verbose.saveLog(obj, folder_log, type=0)
            
            if Verbose.saveFlag:
                
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
            