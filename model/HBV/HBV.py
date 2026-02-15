import re
import subprocess
import numpy as np
import os

from UQPyL.problem import Problem
from UQPyL.util import r_square, mse

class HBV(Problem):
    def __init__(self, projectPath, paraFile, obsFile, exeName):
        
        self.projectPath = projectPath
        self.paraFile = paraFile
        self.exeName = exeName
        self.simDataPath = os.path.join(projectPath, 'rout.dat')
        
        self.xLists, self.xLines, ub, lb = self.readParaFile(paraFile)
        self.obsData = self.readObsFile(obsFile)
        
        nInput = len(self.xLists)
        nOutput = 1
        
        super().__init__(nInput=nInput, nOutput=nOutput, ub=ub, lb=lb, objFunc=self.objFunc, optType='min', xLabels = self.xLists, name = 'HBV')
    
    def readParaFile(self, paraFile):
        
        xLists = []
        xLines = []
        ub = []
        lb = []
        
        with open(paraFile, 'r') as file:
            lines = file.readlines()
        
        for line in lines[1:]:
            name, u, l = line.strip().split()
            
            xLists.append(name)
            ub.append(float(u))
            lb.append(float(l))

        with open(os.path.join(self.projectPath, 'parbas.dat'),'r') as f:
            
            lines = f.readlines()
            
            for name in xLists:
                for i, line in enumerate(lines): 
                    lineSplits = line.strip().split()
                    if len(lineSplits) > 3:
                        if name in lineSplits[3]:
                            xLines.append(i)
                            break
                
        return xLists, xLines, ub, lb
    
    def readObsFile(self, obsFile):
        
        obsData = np.loadtxt(obsFile, dtype=np.float64)
        
        return obsData.reshape(-1, 1)
    
    def setParaVal(self, vals):
        
        with open(os.path.join(self.projectPath, 'parbas.dat'),'r') as f:    
            lines = f.readlines()
        
        for l, v in zip(self.xLines, vals):
            lines[l] = lines[l] = lines[l][:13] + f"{v:>8.5f}"[:8] + lines[l][21:]
        
        with open(os.path.join(self.projectPath, 'parbas.dat'),'w') as f:
            f.writelines(lines)
        
    def objFunc(self, V):
        
        objs = np.zeros((V.shape[0], 1))
        obs = self.obsData
        
        for i, val in enumerate(V):
            
            self.setParaVal(val)
            
            subprocess.run(
                ["./" + self.exeName],
                cwd=self.projectPath,
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL,
                check=True
            )
            
            simData = np.loadtxt(self.simDataPath, dtype=np.float64)[:, 4:5]
                        
            obj = np.sqrt(mse(obs, simData)) #rmse
            
            objs[i, 0] = obj
        
        return objs
            
            
            
            
            
            
        
            
            
            
            
            
                
                
        
        