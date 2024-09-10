import numpy as np
import h5py 

from .population import Population
from .metric import HV, IGD
class Result():
    
    def __init__(self, algorithm):
        
        self.bestDec=None
        self.bestObj=None
        self.appearFEs=None
        self.appearIters=None
        self.historyBestDecs={}
        self.historyBestObjs={}
        self.historyDecs={}
        self.historyObjs={}
        self.historyFEs={}
        self.algorithm=algorithm
        self.historyBestMetrics={}
        
    def update(self, pop: Population, FEs, iter, type=0):
    
        decs=np.copy(pop.decs); objs=np.copy(pop.objs)
        if type==0:
            
            if self.bestObj==None or np.min(objs)<self.bestObj:
                ind=np.where(objs==np.min(objs))
                self.bestDec=decs[ind[0][0], :]
                self.bestObj=objs[ind[0][0], :]
                self.appearFEs=FEs
                self.appearIters=iter
                
            self.historyFEs[FEs]=iter
            self.historyDecs[FEs]=decs
            self.historyObjs[FEs]=objs
            self.historyBestDecs[FEs]=self.bestDec
            self.historyBestObjs[FEs]=self.bestObj
            
        else:
            
            bests=pop.getBest()
            optimum=self.algorithm.problem.getOptimum()
            
            igdValue=None; hvValue=None
            if optimum is not None:
                optimum=optimum[~np.isnan(optimum).any(axis=1)]
                igdValue=IGD(bests, optimum)
            
            hvValue=HV(bests)
            self.historyDecs[FEs]=pop.decs
            self.historyObjs[FEs]=pop.objs
            self.historyBestDecs[FEs]= bests.decs
            self.historyBestMetrics[FEs]= [(hvValue, igdValue) if igdValue is not None else (hvValue)]
            self.historyBestObjs[FEs]= bests.objs
            self.historyFEs[FEs]=iter
            self.bestDec=bests.decs
            self.bestObj=bests.objs
            self.bestMetric=(hvValue, igdValue) if igdValue is not None else (hvValue)
            self.appearFEs=FEs
            self.appearIters=iter

    def save(self, type=0):
        
        import os
        import re
        
        folder=os.path.join(os.getcwd(), "Result")
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        folder_data=os.path.join(folder, "Data")
        if not os.path.exists(folder_data):
            os.mkdir(folder_data)
        
        filename = f"{self.algorithm.name}_{self.algorithm.problem.name}_D{self.algorithm.problem.nInput}_M{self.algorithm.problem.nInput}"
        
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
        self.saveName=filename[:-4]
        
        filepath = os.path.join(folder_data, filename)
        
        historyPopulation={}
        
        digit=len(str(abs(self.algorithm.iters)))
        
        for key in self.historyDecs.keys():
            
            decs=self.historyDecs[key]
            objs=self.historyObjs[key]
            iter=self.historyFEs[key]
            
            item={"FEs" : key , "Decisions" : decs, "Objectives" : objs}

            historyPopulation[f"iter "+str(iter).zfill(digit)]=item
        
        historyBest={}
        for key in self.historyBestDecs.keys():
            
            bestDecs=self.historyBestDecs[key]
            bestObjs=self.historyBestObjs[key]
            iter=self.historyFEs[key]
            
            if type==0:
                item={"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs}
            else:
                metrics=self.historyBestMetrics[key]
                if isinstance(metrics[0], tuple):
                    item={"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs, "HV": metrics[0][0], "IGD": metrics[0][1]}
                else:
                    item={"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs, "HV": metrics}
                    
            historyBest[f"iter "+str(iter).zfill(digit)]=item
        
        globalBest={}
        globalBest["Best Decisions"]=self.bestDec
        globalBest["Best Objectives"]=self.bestObj
        globalBest["FEs"]=self.appearFEs
        globalBest["Iter"]=self.appearIters
        
        if type==1:
            if isinstance(self.bestMetric, tuple):
                globalBest["HV"]=self.bestMetric[0]
                globalBest["IGD"]=self.bestMetric[1]
            else:
                globalBest["HV"]=self.bestMetric
        
        result={ "History_Population" : historyPopulation,
                 "History_Best" : historyBest,
                 "Global_Best" : globalBest,
                 "Max_Iter" : self.algorithm.iters,
                 "Max_FEs" : self.algorithm.FEs }
        
        with h5py.File(filepath, 'w') as f:
            
            save_dict_to_hdf5(f, result)
        
    def reset(self):
        
        self.bestDec=None; self.bestObj=None
        self.appearFEs=None; self.appearIters=None
        self.historyBestDecs={}; self.historyBestObjs={}
        self.historyDecs={}; self.historyObjs={}
        self.historyFEs={}; self.historyMetrics={}

def save_dict_to_hdf5(h5file, d):
    for key, value in d.items():
        if isinstance(value, dict):
            # 如果值是字典，创建一个新的组
            group = h5file.create_group(str(key))
            save_dict_to_hdf5(group, value)  # 递归调用存储子字典
        elif isinstance(value, np.ndarray):
            # 如果值是Numpy数组，直接创建数据集
            h5file.create_dataset(key, data=value)
        else:
            # 如果是标量，保存为一维数组
            h5file.create_dataset(key, data=np.array(value))