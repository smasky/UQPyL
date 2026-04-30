from typing import Optional, Union

import numpy as np


class SpaceBase:
    def __init__(self, nInput: int, xLabels: Optional[list] = None):
        self.nInput = nInput
        if xLabels is None:
            self.xLabels = ["x_" + str(i) for i in range(1, nInput + 1)]
        else:
            self.xLabels = xLabels

    def validate(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] != self.nInput:
            raise ValueError("The input dimension is inconsistent with nInput.")
        return X

    def transform(self, X):
        return self.validate(X)


class Space(SpaceBase):
    def __init__(
        self,
        nInput: int,
        ub: Union[int, float, list, np.ndarray],
        lb: Union[int, float, list, np.ndarray],
        varType: Optional[list] = None,
        varSet: Optional[dict] = None,
        xLabels: Optional[list] = None,
    ):
        super().__init__(nInput=nInput, xLabels=xLabels)

        self._set_ub_lb(ub, lb)

        if varType is None:
            self.varType = np.zeros(self.nInput)
            self.idxF = np.arange(self.nInput)
            self.idxI = np.array([])
            self.idxD = np.array([])
        else:
            if len(varType) != nInput:
                raise ValueError("The length of varType is not equal to nInput.")
            self.varType = np.array(varType, dtype=np.int32)
            self.idxF = np.where(self.varType == 0)[0]
            self.idxI = np.where(self.varType == 1)[0]
            self.idxD = np.where(self.varType == 2)[0]

        if varSet is None:
            self.varSet = {}
        else:
            self.varSet = {}
            for i in self.idxD:
                if i not in varSet:
                    raise ValueError("Missing varSet definition for discrete variable.")
                if isinstance(varSet[i], list):
                    self.varSet[i] = varSet[i]
                else:
                    raise ValueError("The type of sub varSet must be list.")

    def validate(self, X):
        return super().validate(X)

    def transform(self, X):
        X = self.validate(X)
        if self.idxI.size != 0 or self.idxD.size != 0:
            X = self.apply_var_type(X)
        return X

    @property
    def encoding(self):
        return "mix" if (self.idxI.size != 0 or self.idxD.size != 0) else "real"

    def map_discrete_vars(self, X):
        X = self.validate(X).copy()
        if self.idxD.size != 0:
            for i in self.idxD:
                S = self.varSet[i]
                num_interval = len(S)
                bins = np.linspace(self.lb[0, i], self.ub[0, i], num_interval + 1)
                indices = np.digitize(X[:, i], bins, right=False) - 1
                indices[indices == num_interval] = num_interval - 1
                X[:, i] = np.array([S[j] for j in indices])
        return X

    def cast_int_vars(self, X):
        X = self.validate(X).copy()
        if self.idxI.size != 0:
            X[:, self.idxI] = np.round(X[:, self.idxI])
        return X

    def apply_var_type(self, X, IFlag=True, DFlag=True):
        X = self.validate(X).copy()
        if IFlag:
            X = self.cast_int_vars(X)
        if DFlag:
            X = self.map_discrete_vars(X)
        return X

    def unit_to_space(self, X, IFlag=True, DFlag=True):
        X = self.validate(X)
        X_scaled = X * (self.ub - self.lb) + self.lb
        if self.idxI.size != 0 or self.idxD.size != 0:
            X_scaled = self.apply_var_type(X_scaled, IFlag=IFlag, DFlag=DFlag)
        return X_scaled

    # Compatibility wrappers retained during migration.
    def _transform_discrete_var(self, X):
        return self.map_discrete_vars(X)

    def _transform_int_var(self, X):
        return self.cast_int_vars(X)

    def _transform_to_I_D(self, X, IFlag=True, DFlag=True):
        return self.apply_var_type(X, IFlag=IFlag, DFlag=DFlag)

    def _transform_unit_X(self, X, IFlag=True, DFlag=True):
        return self.unit_to_space(X, IFlag=IFlag, DFlag=DFlag)

    def _set_ub_lb(self, ub, lb):
        if isinstance(ub, (int, float)):
            self.ub = np.ones((1, self.nInput)) * ub
        elif isinstance(ub, np.ndarray):
            self._check_bound(ub)
            self.ub = np.atleast_2d(ub)
        elif isinstance(ub, list):
            self.ub = np.atleast_2d(ub)
            self._check_bound(self.ub)
        else:
            raise ValueError("The type of ub is not supported.")

        if isinstance(lb, (int, float)):
            self.lb = np.ones((1, self.nInput)) * lb
        elif isinstance(lb, np.ndarray):
            self._check_bound(lb)
            self.lb = np.atleast_2d(lb)
        elif isinstance(lb, list):
            self.lb = np.atleast_2d(lb)
            self._check_bound(self.lb)
        else:
            raise ValueError("The type of lb is not supported.")

    def _check_bound(self, bound: np.ndarray):
        bound = bound.ravel()
        if bound.shape[0] != self.nInput:
            raise ValueError("The input bound is inconsistent with the nInput of the problem setting")
