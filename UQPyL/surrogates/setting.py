import numpy as np

class Setting():
    
    def __init__(self):
        
        self.parasType = {}
        self.parasSet = {}
        
        self.parasValue = {}
        self.parasUb = {}
        self.parasLb = {}
        self.parasLog = {}
        
        self.parasConst = {}
    
    def setPara(self, name, value, attr = None):
        
        if attr is not None:
            lb, ub, T, S, log = self._check_attr__(attr)
            
            value = np.array([value]) if not isinstance(value, np.ndarray) else value.ravel()
            lb = np.array([lb]) if not isinstance(lb, np.ndarray) else lb.ravel()
            ub = np.array([ub]) if not isinstance(ub, np.ndarray) else ub.ravel()

            if log:
                value = np.log(value); lb = np.log(lb); ub = np.log(ub)
                
            if T != 1:
                value = value.astype(np.float64)
                lb = lb.astype(np.float64)
                ub = ub.astype(np.float64)
            else:
                value = value.astype(np.int32)
                lb = lb.astype(np.int32)
                ub = ub.astype(np.int32)
            
            self.parasValue[name] = value
            self.parasUb[name] = ub
            self.parasLb[name] = lb
            self.parasType[name] = T
            self.parasSet[name] = S
            self.parasLog[name] = log
        else:
            self.parasConst[name] = value
            
    def _check_attr__(self, attr):
        
        if hasattr(attr, 'lb'):
            lb = attr['lb']
        else:
            lb = 0.0
            
        if hasattr(attr, 'ub'):
            ub = attr['ub']
        else:
            ub = 1.0
            
        if hasattr(attr, 'type'):
            T = attr['type']
            if T == 'int':
                T = 1
            elif T == 'float':
                T = 0
            else:
                T = 2
        else:
            T = 0

        if hasattr(attr, 'log'):
            log = attr['log']
        else:
            log = False
            
        if hasattr(attr, 'S'):
            items = attr['S']
            interval = len(items)
            bins = np.linspace(lb[0], ub[0], interval+1)
            S = (items, bins)
        else:
            S = None
        
        return lb, ub, T, S, log            
            
    #Abandoned
    # def setPara(self, name, value, lb, ub, T = 0, S = None, log = False):
        
    #     '''
    #         set parameters to setting for optimization
    #     '''
        
    #     value = np.array([value]) if not isinstance(value, np.ndarray) else value.ravel()
    #     lb = np.array([lb]) if not isinstance(lb, np.ndarray) else lb.ravel()
    #     ub = np.array([ub]) if not isinstance(ub, np.ndarray) else ub.ravel()
        
    #     if T == 0:
    #         lb = lb.astype(np.float64)
    #         ub = ub.astype(np.float64)
    #         value = value.astype(np.float64)
    #     elif T == 1:
    #         lb = lb.astype(np.int32)
    #         ub = ub.astype(np.int32)
    #         value = value.astype(np.int32)
    #     elif T == 2:
    #         lb = np.ones_like(lb)*1e-6
    #         ub = np.ones_like(ub)
            
    #     self.parasValue[name] = value
    #     self.parasUb[name] = ub
    #     self.parasLb[name] = lb
    #     self.parasType[name] = T
    #     self.parasSet[name] = S
    #     self.parasLog[name] = log
        
    # def setPara(self, name, value):
        
    #     self.parasConst[name] = value
      
    def getParaInfos(self, nameLists):
        
        paraInfos = {}
        I = 0
        ub = []
        lb = []
        
        for name in nameLists:
            
            length = self.parasValue[name].size
            paraInfos[name] = np.arange(I, I+length)
            I += length

            if self.parasLog[name]:
                ub.append(np.log(self.parasUb[name]))
                lb.append(np.log(self.parasLb[name]))
            else:
                ub.append(self.parasUb[name])
                lb.append(self.parasLb[name])
            
        return paraInfos, np.concatenate(ub), np.concatenate(lb)
    
    def removeSetting(self, setting):
        
        self.parasValue.pop(setting.parasValue.keys())
        self.parasUb.pop(setting.parasUb.keys())
        self.parasLb.pop(setting.parasLb.keys())
        self.parasSet.pop(setting.parasSet.keys())
        self.parasType.pop(setting.parasType.keys())
    
    def mergeSetting(self, setting):
        
        self.parasValue.update(setting.parasValue)
        self.parasUb.update(setting.parasUb)
        self.parasLb.update(setting.parasLb)
        self.parasSet.update(setting.parasSet)
        self.parasType.update(setting.parasType)
        
    def setValues(self, paraInfos, values):
        
        for name, idx in paraInfos.items():

            if self.parasLog[name]:
                self.parasValue[name][:] = np.exp(values[idx])
            else:
                self.parasValue[name][:] = values[idx]

    def getValues(self, *args):
        
        values=[]
        
        for arg in args:
            if self.parasType[arg]==0:
                values.append(self.parasValue[arg])
            elif self.parasType[arg]==1:
                values.append(self.parasSet[arg].astype(np.int32))
            elif self.parasType[arg]==2:
                #TODO
                S, bins = self.parasSet[arg]
                value = self.parasValue[arg]
                I = np.digitize(value, bins, right=True) - 1
                values.append(S[I])
                
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]