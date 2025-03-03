import numpy as np

class Setting():
    
    def __init__(self, prefix = None):
        
        self.parasType = {}
        self.parasSet = {}
        
        self.parasValue = {}
        self.parasUb = {}
        self.parasLb = {}
        
        self.prefix = prefix
        
    def getParaInfos(self, nameLists):
        
        paraInfos = {}
        I = 0
        ub = []
        lb = []
        T = []
        
        for name in nameLists:
            
            length = self.parasValue[name].size
            paraInfos[name] = np.arange(I, I+length)
            I += length

            ub.append(self.parasUb[name])
            lb.append(self.parasLb[name])
            T.append(self.parasType[name])
            
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
        
    def assignValues(self, paraInfos, values):
        
        for name, idx in paraInfos.items():
            
            self.parasValue[name][:] = values[idx]
            
    def setPara(self, name, value, lb, ub, T = 0, S = None):
        
        '''
            set parameters to setting
        '''
        
        value = np.array([value]) if not isinstance(value, np.ndarray) else value.ravel()
        lb = np.array([lb]) if not isinstance(lb, np.ndarray) else lb.ravel()
        ub = np.array([ub]) if not isinstance(ub, np.ndarray) else ub.ravel()
        
        if T == 0:
            lb = lb.astype(np.float64)
            ub = ub.astype(np.float64)
            value = value.astype(np.float64)
        else:
            lb = lb.astype(np.int32)
            ub = ub.astype(np.int32)
            value = value.astype(np.int32)
            
        self.parasValue[name] = value
        self.parasUb[name] = ub
        self.parasLb[name] = lb
        self.parasType[name] = T
        
        if S is not None:
            num_interval = len(S)
            bins = np.linspace(lb[0], ub[0], num_interval+1)
            self.parasSet[name] = (S, bins)
        else:   
            self.parasSet[name] = (S, None)
        
    def getPara(self, *args):
        '''
         get parameter values from setting
        '''
        
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