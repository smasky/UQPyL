import numpy as np

from .es import ES


class IES(ES):
    """
    Iterative Ensemble Smoother.

    Examples:
        >>> from UQPyL.calibration import IES
        >>> ies = IES(maxIters=5, lam=1e-6, metric="rmse", verboseFlag=False)
        >>> result = ies.run(problem, X)
        >>> print(result.history.metricsHistory)

    References:
        [1] Y. Chen and D. S. Oliver, Iterative ensemble smoother for
            data assimilation, Journal of Petroleum Science and Engineering,
            134:4-15, 2015.

    Notes:
        - `lam` is an explicit regularization parameter added to
          `C_yy + R + lam I`.
        - The configured metric is used for iteration summaries and final
          best-sample selection, not for the update equation itself.
    """

    name = "IES"

    def __init__(
        self,
        verboseFlag: bool = False,
        verboseFreq: int = 1,
        saveFlag: bool = False,
        logFlag: bool = False,
        maxIters: int = 5,
        lam: float = 0.0,
        metric="rmse",
    ):
        super().__init__(
            verboseFlag=verboseFlag,
            verboseFreq=verboseFreq,
            saveFlag=saveFlag,
            logFlag=logFlag,
            metric=metric,
        )
        self.set("maxIters", maxIters)
        self.set("lam", lam)

    def _runCore(self, problem, X, r: np.ndarray | None = None):
        """
        Run iterative ensemble smoothing.

        Args:
            problem: `ModelProblem` instance.
            X: Initial ensemble matrix with shape `(n_samples, n_input)`.
            r: Optional observation error covariance matrix with shape
                `(n_valid_obs, n_valid_obs)`.
        """
        X_cur = np.atleast_2d(X).astype(float, copy=True)
        maxIters = self.get("maxIters")
        lam = self.get("lam")

        priorMean = np.mean(X_cur, axis=0)
        priorScores = None

        for iterIdx in range(maxIters):
            X_next, scores, meta = self._update_once(X_cur, r=r, lam=lam)
            if priorScores is None:
                priorScores = meta["priorScores"]

            self.state.history.metricsHistory.append(
                {
                    "iter": iterIdx + 1,
                    "scoreMean": float(np.mean(scores)),
                }
            )
            X_cur = X_next

        sim_post = self.problem.simFunc(X_cur)
        sim_post_flat = self.problem.flattenSim(sim_post)
        postMean = np.mean(X_cur, axis=0)
        scores = self.score(sim_post_flat)
        bestIdx = int(np.argmin(scores))

        self.recordPosterior(X_cur.copy(), sim_post_flat.copy())
        self.recordBest(X_cur[bestIdx:bestIdx + 1], sim_post_flat[bestIdx:bestIdx + 1])
        self.state.diagnostics["priorMean"] = priorMean
        self.state.diagnostics["posteriorMean"] = postMean
        self.state.diagnostics["priorScores"] = priorScores
        self.state.diagnostics["scores"] = scores
        self.state.extra["bestIdx"] = bestIdx

    def _update_once(self, X, r: np.ndarray | None = None, lam: float = 0.0):
        """
        Apply one iterative smoother update.

        Args:
            X: Current ensemble matrix.
            r: Observation error covariance matrix.
            lam: Explicit diagonal regularization strength.

        Returns:
            tuple: `(X_post, scores, meta)` for the updated ensemble.
        """
        X = np.atleast_2d(X).astype(float, copy=False)
        Y = self.evaluate(X, validOnly=True)
        obs = self.getValidObs().astype(float, copy=False)
        priorScores = self.score(self.evaluate(X, validOnly=False))

        if r is None:
            r = np.zeros((obs.size, obs.size), dtype=float)
        else:
            r = np.asarray(r, dtype=float)

        if r.shape != (obs.size, obs.size):
            raise ValueError("Observation error covariance R must have shape (n_valid_obs, n_valid_obs).")

        x_mean = np.mean(X, axis=0, keepdims=True)
        y_mean = np.mean(Y, axis=0, keepdims=True)
        dX = X - x_mean
        dY = Y - y_mean

        n_ens = X.shape[0]
        if n_ens < 2:
            raise ValueError("IES requires at least two ensemble members.")

        scale = 1.0 / (n_ens - 1)
        c_xy = dX.T @ dY * scale
        c_yy = dY.T @ dY * scale
        gain = c_xy @ np.linalg.inv(c_yy + r + lam * np.eye(obs.size, dtype=float))

        innovation = obs.reshape(1, -1) - Y
        X_post = X + innovation @ gain.T

        sim_post = self.problem.simFunc(X_post)
        sim_post_flat = self.problem.flattenSim(sim_post)
        scores = self.score(sim_post_flat)

        return X_post, scores, {"priorScores": priorScores}
