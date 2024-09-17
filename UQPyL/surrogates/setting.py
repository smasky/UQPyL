import numpy as np

class Setting():
    
    def __init__(self, prefix= None):
        
        self.paras={}
        self.paras_ub={}
        self.paras_lb={}
        self.prefix=prefix
    
    def getParaInfos(self):
        
        position={}; valueList=[]; ubList=[]; lbList=[]; x_labels=[]
        
        pos=0
        for key, item in self.paras.items():
            if isinstance(item, dict):
                for subKey, subItem in item.items():
                    if subItem.size>1:
                        x_labels+=[f"{key}.{subKey}_{i}" for i in range(subItem.size)]
                        index=np.arange(pos, pos+subItem.size)
                        pos=pos+subItem.size
                    else:
                        x_labels+=[f"{key}.{subKey}"]
                        index=np.arange(pos, pos+1)
                        pos+=1
                    valueList+=item.values()
                    ubList.append(self.paras_ub[key][subKey])
                    lbList.append(self.paras_lb[key][subKey])
                    position[f"{key}.{subKey}"]=index
            else:
                x_labels.append(key)
                valueList.append(item)
                ubList.append(self.paras_ub[key])
                lbList.append(self.paras_lb[key])
                position[f"{key}"]=np.arange(pos, pos+1)
                pos+=1
        
        if len(x_labels)>1:  
            return x_labels, np.concatenate(valueList), np.concatenate(ubList), np.concatenate(lbList), position
        else:
            return x_labels, valueList[0], ubList[0], lbList[0], position
        
    def addSubSetting(self, setting):
        
        prefix=setting.prefix
        self.paras[prefix]=setting.paras
        self.paras_lb[prefix]=setting.paras_lb
        self.paras_ub[prefix]=setting.paras_ub
        
    def assignValues(self, position, values):
        
        for key, index in position.items():
            lists=key.split('.')
            value=values[index]
            value=np.array(value).ravel() if not isinstance(value, np.ndarray) else value.ravel()
            
            if len(lists)==1:
                self.paras[lists[0]]=value
            else:
                self.paras[lists[0]][lists[1]]=value
                
    def setPara(self, key, value, lb, ub):
        
        value=np.array(value).ravel() if not isinstance(value, np.ndarray) else value.ravel()
        self.paras[key]=value
        
        lb=np.array(lb).ravel() if not isinstance(lb, np.ndarray) else lb.ravel()
        self.paras_lb[key]=lb
        
        ub=np.array(ub).ravel() if not isinstance(ub, np.ndarray) else ub.ravel()
        self.paras_ub[key]=ub
        
    def getPara(self, *args):
        
        values=[]
        for arg in args:
            lists=arg.split('.')
            if len(lists)==1:
                value=self.paras[lists[0]]
            else:
                value=self.paras[lists[0]][lists[1]]
            value=value[0] if value.size==1 else value
            values.append(value)
            
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]