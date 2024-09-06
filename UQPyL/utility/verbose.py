import numpy as np
import functools
import time
import os
import sys
from datetime import datetime
from prettytable import PrettyTable
            
class Verbose():
    '''
    This is a class for printing and record verbose messages.
    '''
    logFile=None
    logFlag=False
    verbose=False
    workDir=os.getcwd()
    
    @staticmethod    
    def output(obj):
        
        if isinstance(obj, PrettyTable):
            obj=str(obj)+'\n'
        
        if Verbose.logFlag:
            Verbose.logFile.write(obj)
        
        if Verbose.verbose:
            print(obj)
    
    @staticmethod
    def verboseSetting(al):
    
        total_width=os.get_terminal_size().columns
        
        if al.verbose or al.logFlag:
            
            title=al.name+" Setting"
            spacing=int((total_width-len(title))/2)
            Verbose.output("="*spacing+title+"="*spacing)
            keys=al.setting.keys
            values=al.setting.values
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
    def verboseSolutions(dec, obj, x_labels, y_labels, FEs, Iters, width):
        
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
                Verbose.verboseSolutions(obj.result.bestDec, obj.result.bestObj, obj.problem.x_labels, obj.problem.y_labels, obj.FEs, obj.iters, total_width)
        
        return wrapper
    
    @staticmethod
    def decoratorRun(func):
                
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            
            Verbose.logFlag=obj.logFlag
            Verbose.verbose=obj.verbose
            
            if obj.logFlag:
                suffix=datetime.now().strftime("%m%d_%H%M%S")
                Verbose.logFile=open(os.path.join(Verbose.workDir, f"log_{obj.name}_{suffix}.txt"), 'w')
                
            if  obj.verbose or obj.logFlag:
                
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
            
            if obj.verbose:
                title="Conclusion"
                spacing=int((total_width-len(title))/2)
                Verbose.output("="*spacing+title+"="*spacing)
                Verbose.output("Time:  "+Verbose.formatTime(totalTime))
                Verbose.output(f"Used FEs:    {obj.FEs}  |  Iters:  {obj.iters}")
                Verbose.output(f"Best Objs and Best Decision with the FEs")
                Verbose.verboseSolutions(res.bestDec, res.bestObj, obj.problem.x_labels, obj.problem.y_labels, res.appearFEs, res.appearIters, total_width)
            
            if obj.logFlag:
                Verbose.logFile.close()
                
            return res
        return wrapper 
                

            