import numpy as np

class Setting():
    
    def __init__(self):
        self.defaultOwner = None
        
        self.parVal = {}
        self.parCon = {}
        
        self.parUB = {}
        self.parLB = {}
        
        self.parType = {}
        self.parSet = {}
        self.parLog = {}
        self.parOwner = {}
    
    #---------------Public Functions---------------#
    def setPara(self, name, value, attr = None, owner = None):
        '''
        Set the parameter value and its attribute
        :param name: str, the name of the parameter
        :param value: float, int, list, array, the value of the parameter
        :param attr: dict, the attribute of the parameter, including `lb`, `ub`, `type`, `set`, `log`
        :param owner: str, optional owner label such as `model` or `kernel`
        '''
        if owner is None:
            owner = self.defaultOwner

        if owner is not None:
            self.parOwner[name] = owner
        elif name not in self.parOwner:
            self.parOwner[name] = None
        
        if attr is not None:
            lb, ub, T, S, log = self._check_attr__(attr)

            if T == 2 and S is not None:
                value = self._normalize_choice_array(value, S)
            else:
                value = self._normalize_param_array(value, T)
            lb = self._normalize_param_array(lb, T)
            ub = self._normalize_param_array(ub, T)
            
            self.parVal[name] = value
            self.parUB[name] = ub; self.parLB[name] = lb
            self.parType[name] = T; self.parSet[name] = S; self.parLog[name] = log
            
        else:
            self.parCon[name] = value
            
    def getParaInfos(self, nameList):
        '''
        Get the parameter information
        :param nameList: list, the name of the parameter
        :return: tuple, the parameter information, the upper bound and the lower bound
        '''
        paraInfos = {}
        I = 0
        ub = []
        lb = []
        
        for name in nameList:
            if name not in self.parVal:
                raise KeyError(f"Parameter '{name}' is not a tunable parameter.")
            
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
        '''
        Remove the parameter setting
        :param setting: Setting, the setting to be removed
        '''
        for k in list(setting.parVal.keys()):
            self.parVal.pop(k, None)
            self.parUB.pop(k, None)
            self.parLB.pop(k, None)
            self.parSet.pop(k, None)
            self.parType.pop(k, None)
            self.parLog.pop(k, None)
            self.parOwner.pop(k, None)
        for k in list(setting.parCon.keys()):
            self.parCon.pop(k, None)
            self.parOwner.pop(k, None)

    def removeParas(self, nameList):
        for name in list(nameList):
            self.parVal.pop(name, None)
            self.parCon.pop(name, None)
            self.parUB.pop(name, None)
            self.parLB.pop(name, None)
            self.parSet.pop(name, None)
            self.parType.pop(name, None)
            self.parLog.pop(name, None)
            self.parOwner.pop(name, None)

    def removeByOwner(self, owner):
        self.removeParas(self.getParaList(owner=owner, tunableOnly=False))
        
    def mergeSetting(self, setting):
        '''
        Merge the parameter setting
        :param setting: Setting, the setting to be merged
        '''
        self.parVal.update(setting.parVal)
        self.parCon.update(setting.parCon)
        
        self.parUB.update(setting.parUB)
        self.parLB.update(setting.parLB)
        
        self.parSet.update(setting.parSet)
        self.parType.update(setting.parType)
        self.parLog.update(setting.parLog)
        self.parOwner.update(setting.parOwner)

    def hasPara(self, name):
        return name in self.parVal or name in self.parCon

    def getOwner(self, name):
        return self.parOwner.get(name)

    def isChoicePara(self, name):
        return self.parType.get(name) == 2 and self.parSet.get(name) is not None

    def decodeValue(self, name, value):
        if not self.isChoicePara(name):
            return value

        items, bins = self.parSet[name]
        valueArr = np.asarray(value if isinstance(value, (list, tuple, np.ndarray)) else [value], dtype=object).ravel()
        decoded = []

        for item in valueArr:
            if not isinstance(item, (int, float, np.integer, np.floating)):
                decoded.append(item)
                continue

            idx = np.digitize([float(item)], bins, right=True)[0] - 1
            idx = int(np.clip(idx, 0, len(items) - 1))
            decoded.append(items[idx])

        if len(decoded) == 1:
            return decoded[0]

        return decoded

    def getParaList(self, owner = None, tunableOnly = True):
        if tunableOnly:
            names = list(self.parVal.keys())
        else:
            names = list(dict.fromkeys(list(self.parVal.keys()) + list(self.parCon.keys())))

        if owner is None:
            return names

        return [name for name in names if self.parOwner.get(name) == owner]

    def expandParam(self, name, size = None):
        '''
        Materialize a parameter into parVal and expand scalar values to vector form if needed.
        '''
        if name in self.parCon:
            value = self.parCon[name]
        elif name in self.parVal:
            value = self.parVal[name]
        else:
            raise KeyError(f"Parameter '{name}' is not registered.")

        if name not in self.parType:
            raise KeyError(f"Parameter '{name}' is not a tunable parameter.")

        self.parVal[name] = self._normalize_param_array(value, self.parType[name], size=size)

        if name in self.parUB:
            self.parUB[name] = self._normalize_param_array(self.parUB[name], self.parType[name], size=size)

        if name in self.parLB:
            self.parLB[name] = self._normalize_param_array(self.parLB[name], self.parType[name], size=size)

        return self.parVal[name]
    
    def setVals(self, paraInfos, values):
        '''
        Set the parameter value
        :param paraInfos: dict, the parameter information
        :param values: list, the value of the parameter
        '''
        for name, idx in paraInfos.items():
            value = values[idx]

            if self.isChoicePara(name):
                self.parVal[name][:] = self._normalize_choice_array(value, self.parSet[name])
            elif self.parLog[name]:
                self.parVal[name][:] = np.exp(value)
            else:
                self.parVal[name][:] = value
                
    def getVals(self, *args):
        '''
        Get the parameter value
        :param args: list, the name of the parameter
        :return: list, the value of the parameter
        '''
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
        
    #---------------Private Functions---------------#
    def _check_attr__(self, attr):
        '''
        Check the attribute of the parameter
        :param attr: dict, the attribute of the parameter
        :return: tuple, the lower bound, the upper bound, the type and the set
        '''
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

    def _normalize_param_array(self, value, T, size = None):
        value = np.asarray([value] if np.isscalar(value) else value).ravel()

        if size is not None:
            if value.size == 1:
                value = np.repeat(value, size)
            elif value.size != size:
                raise ValueError(f"The dimension of parameter is not consistent with the expected size {size}.")

        if T != 1:
            return value.astype(np.float64)

        return value.astype(np.int32)

    def _normalize_choice_array(self, value, choiceInfo):
        items, bins = choiceInfo
        valueArr = np.asarray(value if isinstance(value, (list, tuple, np.ndarray)) else [value], dtype=object).ravel()
        encoded = np.zeros(valueArr.size, dtype=np.float64)

        for i, item in enumerate(valueArr):
            if isinstance(item, (bool, np.bool_)):
                try:
                    idx = items.index(bool(item))
                except ValueError as exc:
                    raise ValueError(f"Value '{item}' is not in the categorical set.") from exc

                encoded[i] = 0.5 * (bins[idx] + bins[idx + 1])
                continue

            if isinstance(item, (int, float, np.integer, np.floating)):
                encoded[i] = float(item)
                continue

            try:
                idx = items.index(item)
            except ValueError as exc:
                raise ValueError(f"Value '{item}' is not in the categorical set.") from exc

            encoded[i] = 0.5 * (bins[idx] + bins[idx + 1])

        return encoded
            
    def _check_value(self, value, T):
        '''
        Check the value of the parameter
        :param value: float, int, list, array, the value of the parameter
        :param T: int, the type of the parameter
        :return: the value of the parameter
        '''
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
