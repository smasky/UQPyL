import numpy as np
import functools
import os
import sys
import time
from datetime import datetime
from prettytable import PrettyTable

from ..DoE import LHS
from ..problems import ProblemABC

class Population():
    def __init__(self, decs=None, objs=None):
        
        self.decs=np.copy(decs)
        self.objs=np.copy(objs)
        
        if decs is not None:
            self.nPop, self.D=decs.shape
        else:
            self.nPop=0; self.D=0
        if objs is None:
            self.evaluated=None
       
    def __add__(self, otherPop):
        # self.checkSameStatus(otherPop)
        if isinstance(otherPop, np.ndarray):
            return Population(self.decs+otherPop)
        
        return Population(self.decs+otherPop.decs)
    
    def __sub__(self, otherPop):
        
        if isinstance(otherPop, np.ndarray):
            return Population(self.decs-otherPop)
        
        return Population(self.decs-otherPop.decs)
    
    def __mul__(self, number):
        
        return Population(self.decs*number)
    
    def __rmul__(self, number):
        
        return Population(self.decs*number)
    
    def __truediv__(self, number):
        
        return Population(self.decs/number)
    
    def add(self, decs, objs):
        
        otherPop=Population(decs, objs)
        self.add(otherPop)

    def checkSameStatus(self, otherPop):
        
        if self.evaluated != otherPop.evaluated:
            raise Exception("The population evaluation status is different.")
    
    def checkEvaluated(self):
        
        if self.evaluate is False:
            raise Exception("The population is not evaluated yet.")
    
    def initialize(self, decs, objs):
        
        self.decs=decs
        self.objs=objs
        self.nPop, self.D=decs.shape
        
    def getTop(self, k):
        
        args=np.argsort(self.objs.ravel())

        return Population(self.decs[args[:k], :], self.objs[args[:k], :])
    
    def argsort(self):
        
        args=np.argsort(self.objs.ravel())
        
        return args
    
    def clip(self, lb, ub):
        
        self.decs=np.clip(self.decs, lb, ub)
    
    def replace(self, index, pop):
        
        self.decs[index, :]=pop.decs
        self.objs[index, :]=pop.objs
        
    def size(self):
        
        return self.nPop, self.D
    
    def evaluate(self, problem):
        
        self.objs=problem.evaluate(self.decs)
        self.evaluated=True
        
    def add(self, otherPop):
        
        if self.decs is not None:
            self.decs=np.vstack((self.decs, otherPop.decs))
            self.objs=np.vstack((self.objs, otherPop.objs))
        else:
            self.decs=otherPop.decs
            self.objs=otherPop.objs
            
        self.nPop=self.decs.shape[0]
    
    def merge(self, otherPop):
        
        self.add(otherPop)
        
        return self
    
    def __getitem__(self, index):
        
        if isinstance(index, (slice, list, np.ndarray)):
            decs = self.decs[index]
            objs = self.objs[index] if self.objs is not None else None
        elif isinstance(index, (int, np.integer)):
            decs = self.decs[index:index+1]
            objs = self.objs[index:index+1] if self.objs is not None else None
        else:
            raise TypeError("Index must be int, slice, list, or ndarray")
        
        return Population(decs, objs)

    def __len__(self):
        return self.nPop
    
def verboseForUpdate (func):
    
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            
            func(self, *args, **kwargs)
            if self.verbose and self.iters%self.verboseFreq==0:
                total_width=os.get_terminal_size().columns
                title="FEs: "+str(self.FEs)+" | Iters: "+str(self.iters)
                spacing=int((total_width-len(title))/2)
                print("="*spacing+title+"="*spacing)
                verboseSolutions(self.result.bestDec, self.result.bestObj, self.problem.x_labels, self.problem.y_labels, self.FEs, self.iters, total_width)
        
        return wrapper

def verboseSetting(al):
    
    total_width=os.get_terminal_size().columns
    if al.verbose or al.logFlag:
        
        title=al.name+" Setting"
        spacing=int((total_width-len(title))/2)
        print("="*spacing+title+"="*spacing)
        keys=al.setting.keys()
        values=al.setting.values()
        table=PrettyTable(keys)
        table.add_row(values)
        print(table)
    
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
            sys.stdout = Debug(self.verbose, self.logFlag, sys.stdout, self.log_file)
            
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
            verboseSolutions(self.result.bestDec, self.result.bestObj, self.problem.x_labels, self.problem.y_labels, self.result.appearFEs, self.result.appearIters, total_width)
        if self.logFlag:
            self.log_file.close()
        return self.result
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

class Debug(object):
    
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

class Result():
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
        
    def update(self, pop, FEs, Iters):
        
        decs=np.copy(pop.decs); objs=np.copy(pop.objs)
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
    
    def __init__(self, maxFEs, maxIterTimes, maxTolerateTimes=None, tolerate=1e-6, 
                 verbose=True, verboseFreq=10,
                 logFlag=True):
        
        self.setting={}
        self.problem=None
        self.maxFEs=maxFEs
        self.maxIter=maxIterTimes
        self.maxTolerateTimes=maxTolerateTimes
        self.tolerate=tolerate
        self.setting["maxFEs"]=maxFEs
        self.setting["maxIterTimes"]=maxIterTimes
        self.setting["maxTolerateTimes"]=maxTolerateTimes
        
        self.verbose=verbose
        self.verboseFreq=verboseFreq
        self.logFlag=logFlag

        self.result=Result()
    
    def initialize(self):
        
        lhs=LHS('classic', problem=self.problem)
        xInit=lhs.sample(self.nInit, self.problem.n_input)
        pop=Population(xInit)
        self.evaluate(pop); 
            
        return pop
    
    def evaluate(self, pop):
        
        pop.evaluate(self.problem)
        self.FEs+=pop.nPop
    
    def checkTermination(self):
        
        if self.FEs<=self.maxFEs:
            if self.iters<=self.maxIter:
                if self.maxTolerateTimes is None or self.tolerateTimes<=self.maxTolerateTimes:
                    self.iters+=1
                    return True
        return False
    
    @verboseForUpdate
    def record(self, pop):
        
        self.result.update(pop, self.FEs, self.iters)