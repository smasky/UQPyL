from dataclasses import dataclass

import numpy as np
from typing import Optional, Union

from .eval import Eval
from .problem import Problem
from .space import SpaceBase


@dataclass(frozen=True)
class ModelEvalContext:
    sim: np.ndarray
    obs: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


class ModelProblem(Problem):
    """
    Static model problem with a simulation function.

    A `ModelProblem` is a regular `Problem` plus `simFunc`. `simFunc`
    maps batched parameter samples to simulation outputs whose first
    dimension must match the number of input samples.
    """

    def __init__(
        self,
        nInput: int = None,
        nObj: int = 1,
        ub: Union[int, float, np.ndarray, list] = None,
        lb: Union[int, float, np.ndarray, list] = None,
        simFunc: Optional[callable] = None,
        objFunc: Optional[callable] = None,
        conFunc: Optional[callable] = None,
        evaluate: Optional[callable] = None,
        obs: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        conWgt: Optional[list] = None,
        nCon: int = 0,
        varType: list = None,
        varSet: list = None,
        optType: Union[list, str] = 'min',
        xLabels: list = None,
        name: str = None,
        space: Optional[SpaceBase] = None,
        objLabels: list = None,
        conLabels: list = None,
        simLabels: list = None,
    ):
        self._sim_fn = None

        self._validate_model_config(simFunc)
        self._validate_model_callable_config(objFunc, conFunc, evaluate)

        if simFunc is not None:
            self._sim_fn = simFunc

        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        super().__init__(
            nInput=nInput,
            nObj=nObj,
            ub=ub,
            lb=lb,
            objFunc=lambda X: np.zeros((np.atleast_2d(X).shape[0], nObj)),
            nCon=nCon,
            conWgt=conWgt,
            optType=optType,
            varType=varType,
            varSet=varSet,
            xLabels=xLabels,
            space=space,
            objLabels=objLabels,
            conLabels=conLabels,
            name=self.name,
        )

        self._obj_fn = objFunc
        self._con_fn = conFunc
        self._eval_fn = evaluate

        self.obs = None if obs is None else self._validate_obs(obs)
        self.mask = self._validate_mask(mask, self.obs.shape) if self.obs is not None else self._validate_mask_without_obs(mask)
        self.obsShape = None if self.obs is None else self.obs.shape
        self.nObs = None if self.obsShape is None else int(np.prod(self.obsShape))
        self.simLabels = self._validate_sim_labels(simLabels)

    @staticmethod
    def _validate_model_config(simFunc):
        if simFunc is None:
            raise ValueError("`ModelProblem` requires `simFunc`.")

    @staticmethod
    def _validate_model_callable_config(objFunc, conFunc, evaluate):
        if evaluate is not None and (objFunc is not None or conFunc is not None):
            raise ValueError("`evaluate` cannot be used together with `objFunc` or `conFunc`.")
        if conFunc is not None and objFunc is None:
            raise ValueError("`conFunc` cannot be used without `objFunc`.")

    def evaluate(self, X, target=None):
        if target not in (None, "objs", "cons", "sim"):
            raise ValueError("The target must be None, 'objs', 'cons' or 'sim'.")

        X = self.validate(X)
        context = self.buildContext(X)
        if target == "sim":
            return Eval(sim=context.sim)

        eval_fn = getattr(self, "_eval_fn", None)
        if eval_fn is not None:
            eval = eval_fn(X, context)
            if not isinstance(eval, Eval):
                raise TypeError("evaluate must return Eval.")
            objs = eval.objs
            cons = eval.cons
        else:
            objs = self.objFunc(X, context) if target in (None, "objs") else None
            cons = self.conFunc(X, context) if target in (None, "cons") else None

        if target == "objs":
            cons = None
        elif target == "cons":
            objs = None

        return Eval(objs=objs, cons=cons, sim=context.sim)

    def buildContext(self, X):
        sim = self.simFunc(X)
        return ModelEvalContext(sim=sim, obs=self.obs, mask=self.mask)

    def objFunc(self, X, context=None):
        obj_fn = getattr(self, "_obj_fn", None)
        if obj_fn is None:
            raise ValueError("`objFunc` is not defined.")

        if context is None:
            X = self.validate(X)
            context = self.buildContext(X)
        return obj_fn(X, context)

    def conFunc(self, X, context=None):
        con_fn = getattr(self, "_con_fn", None)
        if con_fn is None:
            return None

        if context is None:
            X = self.validate(X)
            context = self.buildContext(X)
        return con_fn(X, context)

    def simFunc(self, X):
        sim_fn = getattr(self, "_sim_fn", None)
        if sim_fn is None:
            raise ValueError("`simFunc` is not defined.")

        X = self.validate(X)
        sim = sim_fn(X)
        return self._validate_sim(sim, X.shape[0])

    def _validate_sim(self, sim, n_samples: int):
        if not isinstance(sim, np.ndarray):
            raise TypeError("Simulation output must be an instance of np.ndarray.")

        if sim.shape[0] != n_samples:
            raise ValueError("Simulation output first dimension must equal n_samples.")

        if not np.issubdtype(sim.dtype, np.number):
            raise TypeError("Simulation output must be numeric.")

        if np.isnan(sim).any():
            raise ValueError("Simulation output must not contain NaN values.")

        return sim

    def flattenSim(self, sim: np.ndarray) -> np.ndarray:
        sim = self._validate_sim(sim, sim.shape[0])
        return sim.reshape(sim.shape[0], -1)

    def flattenObs(self) -> np.ndarray:
        if self.obs is None:
            raise ValueError("Observation `obs` is not defined.")
        return self.obs.reshape(-1)

    def flattenMask(self) -> np.ndarray:
        if self.nObs is None:
            raise ValueError("Observation `obs` is not defined.")
        if self.mask is None:
            return np.zeros(self.nObs, dtype=bool)
        return self.mask.reshape(-1)

    def _validate_obs(self, obs):
        if not isinstance(obs, np.ndarray):
            raise TypeError("Observation `obs` must be an instance of np.ndarray.")
        if obs.ndim != 2:
            raise ValueError("Observation `obs` must be a 2D array with shape (n_time, n_series).")
        if not np.issubdtype(obs.dtype, np.number):
            raise TypeError("Observation `obs` must be numeric.")
        return obs

    def _validate_mask(self, mask, obs_shape):
        if mask is None:
            return None
        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be an instance of np.ndarray.")
        if mask.shape != obs_shape:
            raise ValueError("Mask shape must match obs.shape.")
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        return mask

    def _validate_mask_without_obs(self, mask):
        if mask is not None:
            raise ValueError("Mask requires observation `obs`.")
        return None

    def _validate_sim_labels(self, simLabels):
        if simLabels is None:
            if self.obs is not None:
                return [f"sim_{i}" for i in range(1, self.obs.shape[1] + 1)]
            return None
        return list(simLabels)
