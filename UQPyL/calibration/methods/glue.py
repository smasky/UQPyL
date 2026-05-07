import numpy as np

from ..base import CalibrationABC


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

    def _runCore(self, problem, X, threshold: float):
        """
        Run one GLUE screening pass on a provided sample set.

        Args:
            problem: `ModelProblem` with observations and `simFunc`.
            X: Candidate parameter matrix with shape `(n_samples, n_input)`.
            threshold: Behavioral threshold under the configured metric.
                For metrics where larger is better (for example `nse`),
                behavioral means `score >= threshold`. Otherwise it means
                `score <= threshold`.
        """
        X = np.atleast_2d(X)
        sim_full = self.evaluate(X, validOnly=False)
        scores = self.score(sim_full)
        normalized = self.normalizedScore(sim_full)
        thresholdNorm = -threshold if self.metricHigherIsBetter else threshold
        behavioralMask = normalized <= thresholdNorm

        if not np.any(behavioralMask):
            raise ValueError("No behavioral samples found under the given threshold.")

        bestIdx = int(np.argmin(normalized))
        self.recordBest(X[bestIdx:bestIdx + 1], sim_full[bestIdx:bestIdx + 1])
        self.recordBehavioral(X[behavioralMask], sim_full[behavioralMask])

        self.state.diagnostics["threshold"] = float(threshold)
        self.state.diagnostics["scores"] = scores.copy()
        self.state.diagnostics["behavioralMask"] = behavioralMask.copy()
        self.state.diagnostics["behavioralScores"] = scores[behavioralMask].copy()
