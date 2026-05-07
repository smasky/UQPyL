import numpy as np

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
        This implementation uses a deterministic one-step ensemble smoother
        update in parameter space and evaluates improvement using the
        configured metric.
    """

    name = "ES"

    def __init__(
        self,
        verboseFlag: bool = False,
        verboseFreq: int = 1,
        saveFlag: bool = False,
        logFlag: bool = False,
        metric="rmse",
    ):
        super().__init__(
            verboseFlag=verboseFlag,
            verboseFreq=verboseFreq,
            saveFlag=saveFlag,
            logFlag=logFlag,
            metric=metric,
        )

    def _runCore(self, problem, X, r: np.ndarray | None = None):
        """
        Run a one-step ensemble smoother update.

        Args:
            problem: `ModelProblem` instance.
            X: Ensemble matrix with shape `(n_samples, n_input)`.
            r: Optional observation error covariance matrix with shape
                `(n_valid_obs, n_valid_obs)`. If omitted, zeros are used.
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
            raise ValueError("ES requires at least two ensemble members.")

        scale = 1.0 / (n_ens - 1)
        c_xy = dX.T @ dY * scale
        c_yy = dY.T @ dY * scale
        gain = c_xy @ np.linalg.inv(c_yy + r)

        innovation = obs.reshape(1, -1) - Y
        X_post = X + innovation @ gain.T
        Y_post = self.problem.simFunc(X_post)
        Y_post_flat = self.problem.flattenSim(Y_post)

        post_mean = np.mean(X_post, axis=0)
        scores = self.score(Y_post_flat)
        best_idx = int(np.argmin(scores))

        self.recordPosterior(X_post.copy(), Y_post_flat.copy())
        self.recordBest(X_post[best_idx:best_idx + 1], Y_post_flat[best_idx:best_idx + 1])
        self.state.diagnostics["priorMean"] = np.mean(X, axis=0)
        self.state.diagnostics["posteriorMean"] = post_mean
        self.state.diagnostics["priorScores"] = priorScores
        self.state.diagnostics["scores"] = scores
        self.state.extra["bestIdx"] = best_idx
