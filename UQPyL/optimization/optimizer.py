import numpy as np
import functools
import os
import time
from prettytable import PrettyTable

from ..DoE import LHS

def format_duration(seconds):
    days, seconds = divmod(seconds, 86400)  # 一天有86400秒
    hours, seconds = divmod(seconds, 3600)  # 一小时有3600秒
    minutes, seconds = divmod(seconds, 60)  # 一分钟有60秒
    
    return f"{days} day | {hours} hour | {minutes} minute | {seconds: .2f} second"

def debugDecorator_1(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        total_width=os.get_terminal_size().columns
        if self.verbose:
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
            heads=["FEs"]+self.problem.y_labels+self.problem.x_labels
            bestDec=self.bestDec
            bestObj=self.bestObj
            values=[self.appearFEs]+[ format(item, ".4f") for item in bestObj.ravel()]+[format(item, ".4f") for item in bestDec.ravel()]
            rows=int(len(heads))//10+1
            cols=10
            for i in range(rows):
                if (i+1)*cols<len(heads):
                    end=(i+1)*cols
                else:
                    end=len(heads)
                table=PrettyTable(heads[i*cols:end])
                table.max_width=int(total_width/(cols+4))
                table.min_width=int(total_width/(cols+4))
                table.add_row(values[i*cols:end])
                print(table)
    return wrapper

class Database():
    def __init__(self, problem):
        self.problem=problem
        self.bestDec=None
        self.bestObj=None
        self.appearFEs=None
        self.FEs=0
        self.historyBestDecs={}
        self.historyBestObjs={}
    def update(self, decs, objs):
        if self.bestObj==None or np.min(objs)<self.bestObj:
            ind=np.where(objs==np.min(objs))
            self.database.bestDec=decs[ind[0][0], :]
            self.database.bestObj=objs[ind[0][0], :]
            self.database.appearFEs=self.FEs
            self.database.historyBestDecs[self.FEs]=self.bestDec
            self.database.historyBestObjs[self.FEs]=self.bestObj
            
    def evaluate(self, decs):
        objs=self.problem.evaluate(decs)
        self.FEs+=objs.shape[0]
        return objs
    
class Optimizer():
    FEs=0;maxFEs=0
    iters=0;maxIter=0
    tolerateTimes=0;maxTolerateTimes=None;tolerate=1e-6
    bestDec=None; bestObj=None; appearFEs=None
    historyBestDecs={}; historyBestObjs={}
    def __init__(self, problem, maxFEs, maxIter, maxTolerateTimes, tolerate, verbose):
        self.maxFEs=maxFEs
        self.maxIter=maxIter
        self.maxTolerateTimes=maxTolerateTimes
        self.tolerate=tolerate
        self.verbose=verbose

        self.database=Database(problem)
        
    def evaluate(self, decs):
        return self.database.evaluate(decs)
    
    def checkLoopCriteria(self):
        self.iters+=1
        if self.database.FEs<=self.maxFEs:
            if self.iters<=self.maxIter:
                if self.maxTolerateTimes is None or self.tolerateTimes<=self.maxTolerateTimes:
                    return True
        return False 