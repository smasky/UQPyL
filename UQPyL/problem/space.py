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
        if X.ndim != 2 or X.shape[1] != self.nInput:
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
            types = np.asarray(varType)
            if types.shape != (nInput,):
                raise ValueError("varType must be a vector of length nInput.")
            if types.dtype.kind not in "iuf" or not np.all(np.isin(types, [0, 1, 2])):
                raise ValueError("varType must contain only integer values 0, 1, or 2.")
            self.varType = types.astype(np.int32, copy=True)
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

    def _validate_unit(self, X):
        U = np.asarray(self.validate(X), dtype=float).copy()
        if not np.all(np.isfinite(U)) or np.any(U < -1e-12) or np.any(U > 1+1e-12):
            raise ValueError("Unit coordinates must be finite and within [0, 1].")
        if (not np.all(np.isfinite(self.lb)) or not np.all(np.isfinite(self.ub))
                or np.any(self.ub < self.lb)):
            raise ValueError("Unit conversion requires finite, ordered bounds.")
        return np.clip(U, 0.0, 1.0)

    def _discrete_values(self, index):
        values = np.asarray(self.varSet.get(index, []), dtype=float)
        if (values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values))
                or np.unique(values).size != values.size):
            raise ValueError(f"varSet[{index}] must contain distinct finite numeric values.")
        return values

    def _integer_range(self, index):
        lower, upper = np.ceil(self.lb[0, index]), np.floor(self.ub[0, index])
        if lower > upper:
            raise ValueError(f"Variable {index} has no integer within its bounds.")
        return lower, upper - lower + 1

    def unit_to_space(self, X, IFlag=True, DFlag=True):
        """Decode a unit-cube copy; integer and discrete values use equal bins."""
        U = self._validate_unit(X)
        decoded = np.clip(U * (self.ub - self.lb) + self.lb, self.lb, self.ub)
        if IFlag:
            for index in self.idxI:
                lower, count = self._integer_range(index)
                decoded[:, index] = lower + np.minimum(np.floor(U[:, index]*count), count-1)
        if DFlag:
            for index in self.idxD:
                values = self._discrete_values(index)
                positions = np.minimum(np.floor(U[:, index]*len(values)).astype(int), len(values)-1)
                decoded[:, index] = values[positions]
        return decoded

    def space_to_unit(self, X):
        """Encode real values; integers and discrete choices use bin midpoints."""
        real = np.asarray(self.validate(X), dtype=float)
        if not np.all(np.isfinite(real)):
            raise ValueError("Real coordinates must be finite.")
        self._validate_unit(np.zeros_like(real))
        encoded = np.full(real.shape, 0.5)
        for index in self.idxF:
            lower, upper = self.lb[0, index], self.ub[0, index]
            if np.any(real[:, index] < lower) or np.any(real[:, index] > upper):
                raise ValueError(f"Variable {index} is outside its real bounds.")
            if upper > lower:
                encoded[:, index] = (real[:, index]-lower)/(upper-lower)
        for index in self.idxI:
            lower, count = self._integer_range(index)
            values = real[:, index]
            if (np.any(values != np.floor(values)) or np.any(values < lower)
                    or np.any(values > lower+count-1)):
                raise ValueError(f"Variable {index} must be a legal integer.")
            encoded[:, index] = (values-lower+0.5)/count
        for index in self.idxD:
            choices = self._discrete_values(index)
            matches = real[:, index, None] == choices[None, :]
            if not np.all(np.any(matches, axis=1)):
                raise ValueError(f"Variable {index} must belong to varSet[{index}].")
            encoded[:, index] = (np.argmax(matches, axis=1)+0.5)/len(choices)
        return encoded

    def canonicalize_unit(self, X):
        """Give each integer/discrete real value a unique model input."""
        U = self._validate_unit(X)
        mixed = np.concatenate((self.idxI, self.idxD)).astype(int)
        if mixed.size:
            encoded = self.space_to_unit(self.unit_to_space(U))
            U[:, mixed] = encoded[:, mixed]
        fixed = np.asarray(self.idxF, dtype=int)
        fixed = fixed[self.ub[0, fixed] == self.lb[0, fixed]]
        U[:, fixed] = 0.5
        return U

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
