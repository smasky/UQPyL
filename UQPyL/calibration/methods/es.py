import numpy as np

from ._ensemble import validateCovariance, squareRootAnalysis

from ..base import CalibrationABC


class ES(CalibrationABC):
    """
    Ensemble Smoother with a single deterministic update pass.

    Examples:
        >>> from UQPyL.calibration import ES
        >>> es = ES(metric="rmse", verboseFlag=False)
        >>> result = es.run(problem, X)
        >>> print(result.posteriorDecs.shape)

    References:
        [1] G. Evensen, The ensemble Kalman filter: theoretical formulation
            and practical implementation, Ocean Dynamics, 53:343-367, 2003.
        [2] Y. Chen and D. S. Oliver, Levenberg-Marquardt forms of the
            iterative ensemble smoother for efficient history matching and
            uncertainty quantification, Computational Geosciences, 17:689-703, 2013.

    Notes:
        This implementation uses a symmetric square-root ensemble smoother
        with Kalman mean and covariance before projection, and evaluates the
        configured metric. Continuous updates are projected onto the declared
        box before simulation; diagnostics["boundUpdates"] records adjustments.
        Initial ensembles must be inside the box. Integer/discrete variables
        and general constraints are unsupported.
    """

    name = "ES"
    continuousOnly = True

    def __init__(
        self,
        verboseFlag: bool = False,
        verboseFreq: int = 1,
        saveFlag: bool = False,
        logFlag: bool = False,
        metric="rmse",
        boundHandling: str = "clip",
    ):
        super().__init__(
            verboseFlag=verboseFlag,
            verboseFreq=verboseFreq,
            saveFlag=saveFlag,
            logFlag=logFlag,
            metric=metric,
        )
        if boundHandling not in ("clip", "rescale"):
            raise ValueError("boundHandling must be 'clip' or 'rescale'.")
        self.set("boundHandling", boundHandling)

    def _runCore(self, problem, X, r: np.ndarray | None = None):
        """
        Run a one-step ensemble smoother update.

        Args:
            problem: `ModelProblem` instance.
            X: Ensemble matrix with shape `(n_samples, n_input)`.
            r: Optional observation error covariance matrix with shape
                `(n_valid_obs, n_valid_obs)`. If omitted, zeros are used.
        """
        X = self._validateEnsemble(X)
        obs, r = self._prepareObservations(r)
        fullSim = self.evaluate(X, validOnly=False)
        Y = fullSim[:, self.getValidMask()]
        priorScores = self.score(fullSim)

        analysis, solveInfo = squareRootAnalysis(X, Y, obs, r)
        self.state.diagnostics["covarianceSolve"] = solveInfo
        self.state.diagnostics["updateMethod"] = "symmetric_square_root"
        X_post = self._boundUpdate(analysis, X)
        Y_post = self.problem.simFunc(X_post)
        Y_post_flat = self.problem.flattenSim(Y_post)

        post_mean = np.mean(X_post, axis=0)
        scores = self.score(Y_post_flat)
        bestIdx = int(np.argmin(self.normalizedScore(Y_post_flat)))

        self.recordPosterior(X_post.copy(), Y_post_flat.copy())
        self.recordBest(X_post[bestIdx : bestIdx + 1], Y_post_flat[bestIdx : bestIdx + 1])
        self.state.diagnostics["priorMean"] = np.mean(X, axis=0)
        self.state.diagnostics["posteriorMean"] = post_mean
        self.state.diagnostics["priorScores"] = priorScores
        self.state.diagnostics["scores"] = scores
        self.state.extra["bestIdx"] = bestIdx

    def _prepareObservations(self, r):
        """Validate observation inputs without invoking the simulator."""
        obs = self.getValidObs().astype(float, copy=False)
        if obs.size == 0 or not np.all(np.isfinite(obs)):
            raise ValueError(
                "ES/IES require at least one finite, unmasked observation and no unmasked nonfinite values."
            )
        # None means zero observation noise without allocating a dense zero R.
        return obs, None if r is None else validateCovariance(r, obs.size)

    def _validateEnsemble(self, X):
        """Validate supported parameter domains before calling the simulator."""
        problem = self.problem
        if len(problem.idxI) or len(problem.idxD) or problem.nCon:
            raise NotImplementedError("ES/IES support continuous variables with box bounds only.")
        if problem.lb is None or problem.ub is None:
            raise ValueError("ES/IES require declared box bounds.")
        lower, upper = np.asarray(problem.lb), np.asarray(problem.ub)
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("ES/IES require ordered bounds without NaN.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or X.shape[1] != problem.nInput:
            raise ValueError("Ensemble must have shape (n_samples, n_input).")
        if len(X) < 2:
            raise ValueError(f"{self.name} requires at least two ensemble members.")
        if not np.all(np.isfinite(X)) or np.any(X < lower) or np.any(X > upper):
            raise ValueError("Initial ensemble must be finite and within problem bounds.")
        return X

    def _boundUpdate(self, X, origin=None):
        """Project continuous updates onto the box before simulation."""
        if not np.all(np.isfinite(X)):
            raise ValueError("Ensemble update must be finite before bound projection.")
        bounded = np.clip(X, self.problem.lb, self.problem.ub)
        if self.get("boundHandling") == "rescale" and origin is not None:
            direction = X - origin
            ratios = np.ones_like(X)
            np.divide(self.problem.ub - origin, direction, out=ratios, where=direction > 0)
            np.divide(self.problem.lb - origin, direction, out=ratios, where=direction < 0)
            step = np.minimum(1.0, np.min(ratios, axis=1))
            step = np.where(step < 1, 0.99 * np.maximum(step, 0), step)
            bounded = np.clip(origin + step[:, None] * direction, self.problem.lb, self.problem.ub)
        changed = bounded != X
        self.state.diagnostics.setdefault("boundUpdates", []).append(
            {
                "adjusted_members": int(np.count_nonzero(np.any(changed, axis=1))),
                "adjusted_values": int(np.count_nonzero(changed)),
            }
        )
        self.state.diagnostics.setdefault("boundEffects", []).append(
            {
                "mode": self.get("boundHandling"),
                "adjusted_fraction": float(np.mean(np.any(changed, axis=1))),
                "mean_before": X.mean(axis=0),
                "mean_after": bounded.mean(axis=0),
                "spread_before": np.ptp(X, axis=0),
                "spread_after": np.ptp(bounded, axis=0),
                "unconstrained_moments_preserved": not bool(np.any(changed)),
            }
        )
        return bounded
