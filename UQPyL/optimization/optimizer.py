import numpy as np
import functools
import os
import sys
import time
from datetime import datetime
from prettytable import PrettyTable

from ..DoE import LHS

def verboseForUpdate (func):
    
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            
            func(self, *args, **kwargs)
            if self.verbose and self.iters%self.verboseFreq==0:
                total_width=os.get_terminal_size().columns
                title="FEs: "+str(self.FEs)+" | Iters: "+str(self.iters)
                spacing=int((total_width-len(title))/2)
                print("="*spacing+title+"="*spacing)
                verboseSolutions(self.database.bestDec, self.database.bestObj, self.problem.x_labels, self.problem.y_labels, self.FEs, self.iters, total_width)
        return wrapper
           
def verboseForRun (func):
    
    def format_duration(seconds):
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600) 
        minutes, seconds = divmod(seconds, 60) 
        return f"{days} day | {hours} hour | {minutes} minute | {seconds: .2f} second"
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        total_width=os.get_terminal_size().columns
        if self.logFlag:
            suffix=datetime.now().strftime("%m%d_%H%M%S")
            file=f"log_{self.name}_{suffix}.txt"
            self.log_file = open(file, 'w')
            sys.stdout = Tee(self.verbose, self.logFlag, sys.stdout, self.log_file)
            
        if self.verbose or self.logFlag:
            title=self.name+" Setting"
            spacing=int((total_width-len(title))/2)
            print("="*spacing+title+"="*spacing)
            
            keys=self.setting.keys()
            values=self.setting.values()
            table=PrettyTable(keys)
            table.add_row(values)
            print(table)
        
        startTime=time.time()
        func(self, *args, **kwargs)
        endTime=time.time()
        
        if self.verbose:
            title="Conclusion"
            spacing=int((total_width-len(title))/2)
            print("="*spacing+title+"="*spacing)
            print("Time:  "+format_duration(endTime-startTime))
            print(f"Used FEs:    {self.FEs}  |  Iters:  {self.iters}")
            print(f"Best Objs and Best Decision with the FEs")
            verboseSolutions(self.database.bestDec, self.database.bestObj, self.problem.x_labels, self.problem.y_labels, self.database.appearFEs, self.database.appearIters, total_width)
        if self.logFlag:
            self.log_file.close()
        return self.database
    return wrapper

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
        print(table)

class Tee(object):
    def __init__(self, verbose, logFlag, sysOut, fileOut):
        self.sysOut=sysOut
        self.fileOut=fileOut
        self.logFlag=logFlag
        self.verbose=verbose
        
    def write(self, obj):
        if self.logFlag:
            self.fileOut.write(obj)
            self.fileOut.flush()
        if self.verbose:
            self.sysOut.write(obj)
            self.sysOut.flush()
        
    def flush(self):
        if self.logFlag:
            self.fileOut.flush()
        if self.verbose:
            self.sysOut.flush()

class Database():
    def __init__(self):
        self.bestDec=None
        self.bestObj=None
        self.appearFEs=None
        self.appearIters=None
        self.historyBestDecs={}
        self.historyBestObjs={}
        self.historyDecs={}
        self.historyObjs={}
        self.historyFEs={}
        
    def update(self, decs, objs, FEs, Iters):
        
        if self.bestObj==None or np.min(objs)<self.bestObj:
            ind=np.where(objs==np.min(objs))
            self.bestDec=decs[ind[0][0], :]
            self.bestObj=objs[ind[0][0], :]
            self.appearFEs=FEs
            self.appearIters=Iters
            
        self.historyFEs[FEs]=Iters
        self.historyDecs[FEs]=decs
        self.historyObjs[FEs]=objs
        self.historyBestDecs[FEs]=self.bestDec
        self.historyBestObjs[FEs]=self.bestObj
    
    def reset(self):
        self.bestDec=None; self.bestObj=None
        self.appearFEs=None; self.appearIters=None
        self.historyBestDecs={}; self.historyBestObjs={}
        self.historyDecs={}; self.historyObjs={}
        self.historyFEs={}
    
class Optimizer():
    
    FEs=0;maxFEs=0
    iters=0;maxIter=0
    tolerateTimes=0;maxTolerateTimes=None;tolerate=1e-6
    bestDec=None; bestObj=None; appearFEs=None
    historyBestDecs={}; historyBestObjs={}
    
    def __init__(self, problem, maxFEs, maxIter, maxTolerateTimes, tolerate, 
                 verbose=True, verboseFreq=10,
                 logFlag=True):
        self.maxFEs=maxFEs
        self.maxIter=maxIter
        self.maxTolerateTimes=maxTolerateTimes
        self.tolerate=tolerate
        
        self.verbose=verbose
        self.verboseFreq=verboseFreq
        self.logFlag=logFlag

        self.problem=problem
        self.database=Database()
        
        # self.log_file = open('log.txt', 'w')
        # sys.stdout = Tee(sys.stdout, self.log_file)
    
    def evaluate(self, decs):
        
        objs=self.problem.evaluate(decs)
        self.FEs+=decs.shape[0]
        
        return objs
    
    def checkTermination(self):
        if self.FEs<=self.maxFEs:
            if self.iters<=self.maxIter:
                if self.maxTolerateTimes is None or self.tolerateTimes<=self.maxTolerateTimes:
                    self.iters+=1
                    return True
        return False
    
    @verboseForUpdate
    def update(self, decs, objs):
        self.database.update(decs, objs, self.FEs, self.iters)