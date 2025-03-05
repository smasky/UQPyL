import numpy as np

class Setting():
    
    def __init__(self):
        
        self.parVal = {}
        self.parCon = {}
        
        self.parUB = {}
        self.parLB = {}
        
        self.parType = {}
        self.parSet = {}
        self.parLog = {}
        
    def setPara(self, name, value, attr = None):
        
        if attr is not None:
            lb, ub, T, S, log = self._check_attr__(attr)
            
            value = np.array([value]) if not isinstance(value, np.ndarray) else value.ravel()
            lb = np.array([lb]) if not isinstance(lb, np.ndarray) else lb.ravel()
            ub = np.array([ub]) if not isinstance(ub, np.ndarray) else ub.ravel()
                
            if T != 1:
                value = value.astype(np.float64)
                lb = lb.astype(np.float64)
                ub = ub.astype(np.float64)
            else:
                value = value.astype(np.int32)
                lb = lb.astype(np.int32)
                ub = ub.astype(np.int32)
            
            self.parVal[name] = value
            
            self.parUB[name] = ub; self.parLB[name] = lb
            self.parType[name] = T; self.parSet[name] = S; self.parLog[name] = log
        else:
            self.parCon[name] = value
            
    def _check_attr__(self, attr):
        
        namelist = [ v.lower() for v in attr.keys()]
        
        if 'lb' in namelist:
            lb = attr['lb']
        else:
            lb = 0.0
            
        if 'ub' in namelist:
            ub = attr['ub']
        else:
            ub = 1.0
            
        if 'type' in namelist:
            T = attr['type']
            if T == 'int':
                T = 1
            elif T == 'float':
                T = 0
            else:
                T = 2
        else:
            T = 0

        if 'log' in namelist:
            log = attr['log']
        else:
            log = False
            
        if 'set' in namelist:
            items = attr['set']
            interval = len(items)
            bins = np.linspace(lb, ub, interval+1)
            S = (items, bins)
        else:
            S = None
        
        return lb, ub, T, S, log            
            
    def getParaInfos(self, nameLists):
        
        paraInfos = {}
        I = 0
        ub = []
        lb = []
        
        
        
        for name in nameLists:
            
            length = self.parVal[name].size
            paraInfos[name] = np.arange(I, I+length)
            I += length

            if self.parLog[name]:
                ub.append(np.log(self.parUB[name]))
                lb.append(np.log(self.parLB[name]))
            else:
                ub.append(self.parUB[name])
                lb.append(self.parLB[name])
            
        return paraInfos, np.concatenate(ub), np.concatenate(lb)
    
    def removeSetting(self, setting):
        
        self.parVal.pop(setting.parVal.keys())
        self.parCon.pop(setting.parCon.keys())
        
        self.parUB.pop(setting.parUB.keys())
        self.parLB.pop(setting.parLB.keys())
        
        self.parSet.pop(setting.parSet.keys())
        self.parType.pop(setting.parType.keys())
        self.parLog.pop(setting.parLog.keys())
    
    def mergeSetting(self, setting):
        
        self.parVal.update(setting.parVal)
        self.parCon.update(setting.parCon)
        
        self.parUB.update(setting.parUB)
        self.parLB.update(setting.parLB)
        
        self.parSet.update(setting.parSet)
        self.parType.update(setting.parType)
        self.parLog.update(setting.parLog)
        
    def setVals(self, paraInfos, values):
        
        for name, idx in paraInfos.items():

            if self.parLog[name]:
                self.parVal[name][:] = np.exp(values[idx])
            else:
                self.parVal[name][:] = values[idx]

    def getVals(self, *args):
        
        values = []
        
        for arg in args:
            
            if arg in self.parCon.keys():
                values.append(self.parCon[arg])
            else:
                if self.parType[arg] != 2:
                    values.append(self._check_value(self.parVal[arg], self.parType[arg]))
                else:
                    S, bins = self.parSet[arg]
                    value = self.parVal[arg]
                    I = np.digitize(value, bins, right=True)[0] - 1
                    values.append(S[I])
                
        if len(args) > 1:
            return tuple(values)
        else:
            return values[0]
    
    def _check_value(self, value, T):
        
        if isinstance(value, np.ndarray):
            if T != 1:
                value = value.astype(np.float64)
            else:
                value = value.astype(np.int32)
            value = value.item() if value.size == 1 else value.ravel() #TODO
        else:
            if T != 1:
                value = float(value)
            else:
                value = int(value)
        return value
