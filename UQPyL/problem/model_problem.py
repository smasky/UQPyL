import numpy as np
from typing import Optional, Union

from .base import ProblemBase
from .eval import Eval
from .space import SpaceBase


class ModelProblem(ProblemBase):
    """
    Static model problem for calibration-style workflows.

    A `ModelProblem` maps batched parameter samples to simulation outputs
    aligned with observation space. Observations are stored as a 2D matrix
    with shape `(n_time, n_series)`, and `simFunc` must return a simulation
    tensor with shape `(n_samples, n_time, n_series)`.
    """

    def __init__(
        self,
        nInput: int = None,
        ub: Union[int, float, np.ndarray, list] = None,
        lb: Union[int, float, np.ndarray, list] = None,
        simFunc: Optional[callable] = None,
        obs: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        varType: list = None,
        varSet: list = None,
        xLabels: list = None,
        name: str = None,
        space: Optional[SpaceBase] = None,
        simLabels: list = None,
    ):
        self._sim_fn = None

        self._validate_callable_config(simFunc, obs)

        if simFunc is not None:
            self._sim_fn = simFunc

        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        super().__init__(
            nInput=nInput,
            nObj=1,
            ub=ub,
            lb=lb,
            nCon=0,
            optType='min',
            varType=varType,
            varSet=varSet,
            xLabels=xLabels,
            space=space,
        )

        self.nObj = 0
        self.optType = None
        self.opt = None
        self.objLabels = None
        self.yLabels = None
        self.conLabels = None
        self.obs = self._validate_obs(obs)
        self.mask = self._validate_mask(mask, self.obs.shape)
        self.simLabels = self._validate_sim_labels(simLabels, self.obs.shape[1])
        self.obsShape = self.obs.shape
        del self.nOutput
        self.nObs = int(np.prod(self.obsShape))

    @staticmethod
    def _validate_callable_config(simFunc, obs):
        if simFunc is None:
            raise ValueError("`ModelProblem` requires `simFunc`.")
        if obs is None:
            raise ValueError("`ModelProblem` requires `obs`.")

    def evaluate(self, X, target=None):
        if target not in (None, "sim"):
            raise ValueError("The target must be None or 'sim'.")

        X = self.validate(X)
        sim = self.simFunc(X)
        return Eval(sim=sim)

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

        if sim.ndim != 3:
            raise ValueError(
                "Simulation output must be a 3D array with shape (n_samples, n_time, n_series)."
            )

        if sim.shape[0] != n_samples:
            raise ValueError("Simulation output first dimension must equal n_samples.")

        if sim.shape[1:] != self.obsShape:
            raise ValueError("Simulation output shape after n_samples must match obs.shape exactly.")

        if not np.issubdtype(sim.dtype, np.number):
            raise TypeError("Simulation output must be numeric.")

        if np.isnan(sim).any():
            raise ValueError("Simulation output must not contain NaN values.")

        return sim

    def flattenSim(self, sim: np.ndarray) -> np.ndarray:
        sim = self._validate_sim(sim, sim.shape[0])
        return sim.reshape(sim.shape[0], -1)

    def flattenObs(self) -> np.ndarray:
        return self.obs.reshape(-1)

    def flattenMask(self) -> np.ndarray:
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

    def _validate_sim_labels(self, simLabels, n_series):
        if simLabels is None:
            return [f"sim_{i}" for i in range(1, n_series + 1)]
        if len(simLabels) != n_series:
            raise ValueError("The length of simLabels must equal obs.shape[1].")
        return list(simLabels)
