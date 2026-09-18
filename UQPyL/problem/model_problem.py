from dataclasses import dataclass

import numpy as np
from typing import Optional, Union

from .base import ProblemBase
from .evaluator_base import ModelEvaluatorBase, coerce_model_evaluator
from .eval import Eval
from .model_evaluator import ModelEvaluator
from .simulator_base import SimulatorBase
from .space import SpaceBase


@dataclass(frozen=True)
class SimContext:
    sims: np.ndarray
    obs: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


class ModelProblem(ProblemBase):
    """
    Static model problem with a simulation function.

    `ModelProblem` is the simulation-backed problem mode in UQPyL.
    `simFunc` maps batched parameter samples to simulation outputs whose
    first dimension must match the number of input samples.
    """

    _evalTargets = (None, "objs", "cons", "sims")

    class _CallableSimulator(SimulatorBase):
        def __init__(self, owner):
            self.owner = owner

        def run(self, X):
            sims = self.owner.simFunc(X)
            return SimContext(sims=sims, obs=self.owner.obs, mask=self.owner.mask)

    def __init__(
        self,
        nInput: int = None,
        nObj: int = 1,
        ub: Union[int, float, np.ndarray, list] = None,
        lb: Union[int, float, np.ndarray, list] = None,
        simFunc: Optional[callable] = None,
        objFunc: Optional[callable] = None,
        conFunc: Optional[callable] = None,
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
        evaluator: Optional[ModelEvaluatorBase] = None,
        seriesLabels: list = None,
    ):
        self._sim_fn = None

        self._validate_model_config(simFunc)
        self._validate_model_callable_config(objFunc, conFunc, evaluator)

        if simFunc is not None:
            self._sim_fn = simFunc

        self._obj_fn = objFunc
        self._con_fn = conFunc
        self.evaluator = coerce_model_evaluator(evaluator)
        if self.evaluator is None:
            self.evaluator = ModelEvaluator(objFunc=objFunc, conFunc=conFunc)
        self.simulator = self._CallableSimulator(self)

        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        super().__init__(
            nInput=nInput,
            nObj=nObj,
            ub=ub,
            lb=lb,
            nCon=nCon,
            conWgt=conWgt,
            optType=optType,
            varType=varType,
            varSet=varSet,
            xLabels=xLabels,
            space=space,
            objLabels=objLabels,
            conLabels=conLabels,
        )

        self.obs = None if obs is None else self._validate_obs(obs)
        self.mask = self._validate_mask(mask, self.obs.shape) if self.obs is not None else self._validate_mask_without_obs(mask)
        self.obsShape = None if self.obs is None else self.obs.shape
        self.nObs = None if self.obsShape is None else int(np.prod(self.obsShape))
        self.seriesLabels = self._validate_series_labels(seriesLabels)

    @staticmethod
    def _validate_model_config(simFunc):
        if simFunc is None:
            raise ValueError("`ModelProblem` requires `simFunc`.")

    @staticmethod
    def _validate_model_callable_config(objFunc, conFunc, evaluator):
        if conFunc is not None and objFunc is None:
            raise ValueError("`conFunc` cannot be used without `objFunc`.")
        if evaluator is not None and (objFunc is not None or conFunc is not None):
            raise ValueError("`evaluator` cannot be used together with `objFunc` or `conFunc`.")

    def evaluate(self, X, target=None):
        simContext = self.simulator.run(X)
        return self.evaluator.evaluate(X, simContext, target=target)

    def simulate(self, X):
        return self.simulator.run(self.validate(X))

    def objFunc(self, X, context):
        obj_fn = getattr(self, "_obj_fn", None)
        if obj_fn is None:
            raise ValueError("`objFunc` is not defined.")
        if context is None:
            raise ValueError("`ModelProblem.objFunc()` requires explicit `context`.")
        return obj_fn(X, context)

    def conFunc(self, X, context):
        con_fn = getattr(self, "_con_fn", None)
        if con_fn is None:
            return None
        if context is None:
            raise ValueError("`ModelProblem.conFunc()` requires explicit `context`.")
        return con_fn(X, context)

    def simFunc(self, X):
        sim_fn = getattr(self, "_sim_fn", None)
        if sim_fn is None:
            raise ValueError("`simFunc` is not defined.")

        X = self.validate(X)
        sims = sim_fn(X)
        return self._validate_sim(sims, X.shape[0])

    def _validate_sim(self, sims, n_samples: int):
        if not isinstance(sims, np.ndarray):
            raise TypeError("Simulation output must be an instance of np.ndarray.")

        if sims.ndim == 0 or sims.shape[0] != n_samples:
            raise ValueError("Simulation output first dimension must equal n_samples.")

        if not np.issubdtype(sims.dtype, np.number):
            raise TypeError("Simulation output must be numeric.")

        nan_mask = np.isnan(sims)
        if nan_mask.any():
            # Masked observation positions are allowed to be NaN when the
            # simulation output aligns with the observation grid.
            if self.mask is not None and self.obs is not None and sims.shape[1:] == self.obs.shape:
                allowed = np.broadcast_to(self.mask, sims.shape)
                if np.any(nan_mask & ~allowed):
                    raise ValueError("Simulation output must not contain NaN values outside masked positions.")
            else:
                raise ValueError("Simulation output must not contain NaN values.")

        return sims

    def flattenSim(self, sims: np.ndarray) -> np.ndarray:
        sims = self._validate_sim(sims, sims.shape[0])
        return sims.reshape(sims.shape[0], -1)

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

    def _validate_series_labels(self, seriesLabels):
        if seriesLabels is None:
            if self.obs is not None:
                return [f"series_{i}" for i in range(1, self.obs.shape[1] + 1)]
            return None
        return list(seriesLabels)

    def _validate_eval_result(self, evalRes, X, target):
        objs, cons = self._validate_common_eval_result(evalRes, X)

        nSamples = X.shape[0]
        sims = evalRes.sims
        if target in (None, "sims") and sims is None:
            raise ValueError("`ModelProblem` evaluate() must return `sims`.")
        if target in ("objs", "cons") and sims is not None:
            raise ValueError(f"`target='{target}'` requires `sims` to be None.")
        if sims is not None:
            sims = self._validate_sim(np.asarray(sims), nSamples)

        if target is None and self.nCon > 0 and cons is None:
            raise ValueError("`ModelProblem` evaluate() requires `cons` to be returned when nCon > 0.")

        if target == "sims":
            if objs is not None or cons is not None:
                raise ValueError("`target='sims'` requires `objs` and `cons` to be None.")
        elif target == "objs":
            if objs is None:
                raise ValueError("`target='objs'` requires `objs` to be returned.")
            if cons is not None:
                raise ValueError("`target='objs'` requires `cons` to be None.")
        elif target == "cons":
            if cons is None and self.nCon > 0:
                raise ValueError("`target='cons'` requires `cons` to be returned.")
            if objs is not None:
                raise ValueError("`target='cons'` requires `objs` to be None.")

        return Eval(objs=objs, cons=cons, sims=sims)
