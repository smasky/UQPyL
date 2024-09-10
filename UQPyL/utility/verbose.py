import os
import re
import time
import h5py
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
    
        total_width=os.get_terminal_size().columns
        
        if obj.verbose or obj.logFlag:
            
            title=obj.name+" Setting"
            spacing=int((total_width-len(title))/2)
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
        rows=int(len(heads))//10+1
        cols=10
        for i in range(rows):
            if (i+1)*cols<len(heads):
                end=(i+1)*cols
            else:
                end=len(heads)
            table=PrettyTable(heads[i*cols:end])
            table.max_width=int(width/(cols+4))
            table.min_width=int(width/(cols+4))
            table.add_row(values[i*cols:end])
            
            Verbose.output(table)
    
    @staticmethod
    def verboseSingleSolutions(dec, obj, x_labels, y_labels, FEs, Iters, width):
        
        heads=["FEs"]+["Iters"]+y_labels+x_labels
        values=[FEs, Iters]+[ format(item, ".4f") for item in obj.ravel()]+[format(item, ".4f") for item in dec.ravel()]
        rows=int(len(heads))//10+1
        cols=10
        for i in range(rows):
            if (i+1)*cols<len(heads):
                end=(i+1)*cols
            else:
                end=len(heads)
            table=PrettyTable(heads[i*cols:end])
            table.max_width=int(width/(cols+4))
            table.min_width=int(width/(cols+4))
            table.add_row(values[i*cols:end])
            
            Verbose.output(table)
    
    @staticmethod
    def decoratorRecord(func):
        
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            
            func(obj, *args, **kwargs)
            
            if obj.verbose and obj.iters%obj.verboseFreq==0:
                total_width=os.get_terminal_size().columns
                title="FEs: "+str(obj.FEs)+" | Iters: "+str(obj.iters)
                spacing=int((total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)
                if obj.problem.nOutput==1:
                    Verbose.verboseSingleSolutions(obj.result.bestDec, obj.result.bestObj, obj.problem.x_labels, obj.problem.y_labels, obj.FEs, obj.iters, total_width)
                else:
                    Verbose.verboseMultiSolutions(obj.result.bestDec, obj.result.bestMetric, obj.FEs, obj.iters, total_width)
        return wrapper
    
    @staticmethod
    def saveData(obj, folder_data):
        
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
    def saveLog(obj, folder_log):
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
            
            Verbose.logFlag=obj.logFlag
            Verbose.verbose=obj.verbose
            Verbose.saveFlag=obj.saveFlag
            
            #Check result dir
            if Verbose.logFlag or Verbose.saveFlag:
                
                folder=os.path.join(Verbose.workDir, "Result")
                if not os.path.exists(folder):
                    os.mkdir(folder)
                
                folder_data=os.path.join(folder, "Data")
                folder_log=os.path.join(folder, "Log")
                
                if not os.path.exists(folder_data):
                    os.mkdir(folder_data)
                    
                if not os.path.exists(folder_log):
                    os.mkdir(folder_log)
            
            if Verbose.logFlag:
                  
                Verbose.logLines=[]
                
            if  Verbose.verbose or Verbose.logFlag:
                
                total_width=os.get_terminal_size().columns
                title=obj.name+" Setting"
                spacing=int((total_width-len(title))/2)
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
                spacing=int((total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)
                Verbose.output("Time:  "+Verbose.formatTime(totalTime))
                Verbose.output(f"Used FEs:    {obj.FEs}  |  Iters:  {obj.iters}")
                Verbose.output(f"Best Objs and Best Decision with the FEs")
                
                if obj.problem.nOutput==1:
                    Verbose.verboseSingleSolutions(res.bestDec, res.bestObj, obj.problem.x_labels, obj.problem.y_labels, res.appearFEs, res.appearIters, total_width)
                else:
                    Verbose.verboseMultiSolutions(res.bestDec, res.bestMetric, res.appearFEs, res.appearIters, total_width)

            if Verbose.saveFlag:
                
                Verbose.saveData(obj, folder_data)
                
            if Verbose.logFlag:
                
                Verbose.saveLog(obj, folder_log)
   
            return res
        return wrapper 
    
    @staticmethod
    def decoratorAnalyze(self, func):
        
        def wrapper(obj, *args, **kwargs):
        
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
            