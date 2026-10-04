import numpy as np
import warnings

from ..base import CalibrationABC
from ._uncertainty import likelihoodWeights, weightedQuantiles


class GLUE(CalibrationABC):
    """
    Generalized Likelihood Uncertainty Estimation.

    This implementation uses the configured metric to score each candidate
    sample and keeps samples passing the threshold criterion as behavioral
    samples.

    Examples:
        >>> from UQPyL.calibration import GLUE
        >>> glue = GLUE(metric="rmse", verboseFlag=False)
        >>> result = glue.run(problem, X, threshold=0.5)
        >>> print(result.extra["behavioralDecs"].shape)

    References:
        [1] K. Beven and A. Binley, The future of distributed models:
            Model calibration and uncertainty prediction, Hydrological
            Processes, 6(3):279-298, 1992.
    """

    name = "GLUE"

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

    def _runCore(self, problem, X, threshold: float, logLikelihood=None, interval: float = 0.95):
        """
        Run one GLUE screening pass on a provided sample set.

        Args:
            problem: `ModelProblem` with observations and `simFunc`.
            X: Candidate parameter matrix with shape `(n_samples, n_input)`.
            threshold: Behavioral threshold under the configured metric.
                For metrics where larger is better (for example `nse`),
                behavioral means `score >= threshold`. Otherwise it means
                `score <= threshold`. For `pbias`, use `abs(score) <= threshold`,
                with a finite nonnegative threshold in percentage points.
            logLikelihood: Optional callable (obs, behavioralSim, mask=mask)
                returning one log weight per behavioral sample. None means
                uniform weights, not an inferred observation noise model.
            interval: Central weighted empirical-CDF interval probability.
                Bounds cover unmasked simulation outputs, without adding noise.
        """
        if np.ndim(interval) or not np.isfinite(interval) or not 0 < interval < 1:
            raise ValueError("interval must be a finite scalar between 0 and 1.")
        if logLikelihood is not None and not callable(logLikelihood):
            raise ValueError("logLikelihood must be callable or None.")
        if self.metricClosestToZero:
            threshold = float(threshold)
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("PBIAS threshold must be finite and nonnegative.")
        X = np.atleast_2d(X)
        sim_full = self.evaluate(X, validOnly=False)
        scores = self.score(sim_full)
        normalized = self.normalizedScore(sim_full)
        thresholdNorm = -threshold if self.metricHigherIsBetter else threshold
        behavioralMask = normalized <= thresholdNorm

        if not np.any(behavioralMask):
            raise ValueError("No behavioral samples found under the given threshold.")

        bestIdx = int(np.argmin(normalized))
        self.state.extra["bestIdx"] = bestIdx
        self.recordBest(X[bestIdx : bestIdx + 1], sim_full[bestIdx : bestIdx + 1])
        self.recordBehavioral(X[behavioralMask], sim_full[behavioralMask])

        self.state.diagnostics["threshold"] = float(threshold)
        self.state.diagnostics["scores"] = scores.copy()
        self.state.diagnostics["behavioralMask"] = behavioralMask.copy()
        self.state.diagnostics["behavioralScores"] = scores[behavioralMask].copy()
        behavioralSims = sim_full[behavioralMask]
        weights = likelihoodWeights(self.getFlattenedObs(), behavioralSims, self.getFlattenedMask(), logLikelihood)
        bounds = weightedQuantiles(behavioralSims[:, self.getValidMask()], weights, interval)
        self.state.diagnostics["behavioralWeights"] = weights
        self.state.diagnostics["effectiveSampleSize"] = float(1.0 / np.sum(weights**2))
        lowEss = self.state.diagnostics["effectiveSampleSize"] < 20 * (1 - 64 * np.finfo(float).eps)
        self.state.diagnostics["uncertaintyStatus"] = "low_effective_sample_size" if lowEss else "estimated"
        self.state.diagnostics["weighting"] = "uniform" if logLikelihood is None else "log_likelihood"
        self.state.diagnostics["interval"] = float(interval)
        self.state.diagnostics["ppuLower"] = bounds[0]
        self.state.diagnostics["ppuUpper"] = bounds[1]
        if lowEss and logLikelihood is not None:
            warnings.warn(
                "GLUE uncertainty effective sample size is below 20; weighted intervals may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
