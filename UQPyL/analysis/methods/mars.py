import numpy as np
import warnings
from typing import Optional

from ..base import AnaIndex, AnalysisABC
from ...problem import ProblemABC as Problem
from ...surrogate.mars import MARS as MARSModel
from ._variance import scaleOutput, restoreSquaredOutput


class MARS(AnalysisABC):
    """MARS screening using positive leave-variable-out GCV increases.

    A deterministic holdout checks the full surrogate before reporting scores.
    Scores have squared output units and are not Sobol variance fractions.
    """

    name = "MARS"

    def __init__(
        self,
        verboseFlag: bool = True,
        logFlag: bool = False,
        saveFlag: bool = False,
        *,
        maxDegree: int = 2,
        maxTerms: int = 40,
        minValidationR2: float = 0.8,
        nValidationRepeats: int = 1,
        stabilityTolerance: float = 0.05,
        gcvImprovementTolerance: float = 0.02,
    ):
        """Configure interaction order, basis size and holdout R2 threshold.

        The threshold triggers a predictive-quality warning, not a confidence level.
        At least 20 representative, exchangeable sample rows are required.
        Extra holdouts diagnose variation in normalized importance; primary
        scores always use split seed 0. Stability is not proof of accuracy.
        """
        super().__init__(verboseFlag, logFlag, saveFlag)
        for name, value in (
            ("maxDegree", maxDegree),
            ("maxTerms", maxTerms),
            ("nValidationRepeats", nValidationRepeats),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            self.set(name, value)
        for name, value in (
            ("minValidationR2", minValidationR2),
            ("stabilityTolerance", stabilityTolerance),
            ("gcvImprovementTolerance", gcvImprovementTolerance),
        ):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
            self.set(name, float(value))

    def _analyzeCore(
        self,
        problem: Problem,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
        target: str = "objs",
        index: AnaIndex = "all",
    ) -> None:
        Y = self.check_Y(X, Y, target, index)
        if len(X) < 20:
            raise ValueError("MARS analysis requires at least 20 samples for training and validation.")
        scores, normalized, primaryDiagnostic = self._analyzeSplit(problem, X, Y, 0)
        self.state.extra["mars_validation"] = primaryDiagnostic
        splitSeeds = list(range(self.get("nValidationRepeats")))
        splitWeights = [normalized]
        splitDiagnostics = [dict(**primaryDiagnostic, normalized_weights=normalized.tolist())]
        for splitSeed in splitSeeds[1:]:
            _, weights, diagnostic = self._analyzeSplit(problem, X, Y, splitSeed)
            splitWeights.append(weights)
            splitDiagnostics.append(dict(**diagnostic, normalized_weights=weights.tolist()))
        weights = np.stack(splitWeights)
        assessed = len(splitSeeds) > 1
        stabilityOutputs = []
        for outputIndex in range(Y.shape[1]):
            outputWeights = weights[:, outputIndex, :]
            minimum, maximum = np.min(outputWeights, axis=0), np.max(outputWeights, axis=0)
            maxRange = float(np.max(maximum - minimum))
            stable = maxRange <= self.get("stabilityTolerance") if assessed else None
            if assessed and not stable:
                warnings.warn(
                    f"MARS output {outputIndex} normalized importance range={maxRange:.6g} across holdouts "
                    f"exceeds stabilityTolerance={self.get('stabilityTolerance')}; screening weights are unstable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            stabilityOutputs.append(
                dict(
                    constant_output=primaryDiagnostic["outputs"][outputIndex]["constant_output"],
                    stable=stable,
                    max_normalized_range=maxRange,
                    normalized_min=minimum.tolist(),
                    normalized_max=maximum.tolist(),
                    normalized_mean=np.mean(outputWeights, axis=0).tolist(),
                    normalized_std=np.std(outputWeights, axis=0, ddof=0).tolist(),
                )
            )
        self.state.extra["mars_stability"] = dict(
            n_validation_repeats=len(splitSeeds),
            split_seeds=splitSeeds,
            stability_tolerance=self.get("stabilityTolerance"),
            assessed=assessed,
            outputs=stabilityOutputs,
            splits=splitDiagnostics,
        )
        self.recordResult(
            X,
            Y,
            [
                ("S1", scores, self.outputLabels, problem.xLabels, "decsDim1"),
                ("S1_norm", normalized, self.outputLabels, problem.xLabels, "decsDim1"),
            ],
            target=target,
            meta=meta,
        )

    def _analyzeSplit(self, problem, X, Y, splitSeed):
        # A fixed local split makes repeat runs comparable without changing global RNG.
        order = np.random.default_rng(splitSeed).permutation(len(X))
        validationRows, trainRows = order[: max(5, len(X) // 5)], order[max(5, len(X) // 5) :]
        # Fit preprocessing on training rows only; constant columns remain zero.
        origin = np.min(X[trainRows], axis=0)
        spread = np.ptp(X[trainRows], axis=0)
        scaledX = (X - origin) / np.where(spread > 0, spread, 1.0)
        # Forward search can add two terms past max_terms. Keep GCV effective
        # parameter count (2.5 * basis_size - 1.5 at penalty=3) below training N.
        termLimit = min(self.get("maxTerms"), max(1, int((len(trainRows) - 1) / 2.5) - 2))
        scores = np.zeros((Y.shape[1], problem.nInput))
        normalized = np.zeros_like(scores)
        diagnostics = []
        for outputIndex in range(Y.shape[1]):
            values = Y[:, outputIndex : outputIndex + 1]
            if not np.all(np.isfinite(values)):
                raise ValueError("Sensitivity analysis requires finite output values.")
            if np.all(values == values[0]):
                diagnostics.append(dict(validation_r2=None, constant_output=True))
                continue
            if np.all(values[trainRows] == values[trainRows[0]]):
                raise ValueError("MARS training output is constant but validation output is not.")
            scaledY, outputScale = scaleOutput(values, fitRows=trainRows, returnScale=True)

            def fitModel(inputs):
                # Paired hinge terms give the forward search interaction parents;
                # a knotless additive term can otherwise stop that search early.
                model = MARSModel(
                    max_degree=self.get("maxDegree"),
                    max_terms=termLimit,
                    thresh=1e-6,
                    allow_linear=False,
                )
                model.fit(inputs[trainRows], scaledY[trainRows])
                return model

            model = fitModel(scaledX)
            actual = scaledY[validationRows]
            residual = actual - model.predict(scaledX[validationRows])
            total = float(np.sum((actual - np.mean(actual)) ** 2))
            r2 = 1 - float(np.sum(residual**2)) / total if total > 0 else -np.inf
            if not np.isfinite(r2) or r2 < self.get("minValidationR2"):
                warnings.warn(
                    f"MARS output {outputIndex} split {splitSeed} validation R2={r2:.6g} is below "
                    f"minValidationR2={self.get('minValidationR2')}; importance is unreliable. "
                    "Provide more representative samples or adjust maxDegree/maxTerms.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            baseGcv = model.gcv_
            reducedGcvs = []
            scaledScores = np.zeros(problem.nInput)
            for variable in range(problem.nInput):
                reducedX = np.delete(scaledX, variable, axis=1)
                if reducedX.shape[1] == 0:
                    # Intercept-only model, with the same GCV correction as MARS.
                    reducedGcv = float(np.mean(scaledY[trainRows] ** 2)) / (1 - 1 / len(trainRows)) ** 2
                else:
                    reducedGcv = fitModel(reducedX).gcv_
                if not np.all(np.isfinite([baseGcv, reducedGcv])):
                    raise ValueError("MARS produced non-finite GCV; importance is unreliable.")
                reducedGcvs.append(float(reducedGcv))
                scaledScores[variable] = max(0.0, reducedGcv - baseGcv)
            improvementVariable = int(np.argmin(reducedGcvs))
            improvement = max(0.0, baseGcv - reducedGcvs[improvementVariable])
            baselineMse = float(np.mean(scaledY[trainRows] ** 2))
            improvementFraction = improvement / baselineMse
            # A large improvement after deleting an input can expose a different
            # greedy fit path. Ignore negligible errors in nearly exact fits.
            searchUnstable = bool(
                improvementFraction > self.get("gcvImprovementTolerance") and improvement >= 0.5 * baseGcv
            )
            if searchUnstable:
                warnings.warn(
                    f"MARS output {outputIndex} split {splitSeed} GCV improves substantially when removing "
                    f"input {problem.xLabels[improvementVariable]}; surrogate search is unstable and "
                    "importance may be unreliable even with high validation R2.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            scoreScale = np.max(scaledScores)
            if scoreScale > 0:
                relativeScores = scaledScores / scoreScale
                normalized[outputIndex] = relativeScores / np.sum(relativeScores)
            scores[outputIndex], underflow = restoreSquaredOutput(scaledScores, outputScale, "MARS")
            rawGcvs, _ = restoreSquaredOutput([baseGcv, *reducedGcvs], outputScale, "MARS", warnUnderflow=False)
            diagnostics.append(
                dict(
                    validation_r2=r2,
                    constant_output=False,
                    base_gcv=float(rawGcvs[0]),
                    removed_gcv=rawGcvs[1:].tolist(),
                    scaled_base_gcv=float(baseGcv),
                    scaled_removed_gcv=reducedGcvs,
                    scale_mantissa=outputScale[0],
                    scale_exponent=outputScale[1],
                    raw_underflow=underflow,
                    gcv_improvement_fraction=float(improvementFraction),
                    gcv_improvement_variable=improvementVariable if improvement > 0 else None,
                    gcv_search_unstable=searchUnstable,
                )
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("MARS importance exceeds the finite output range.")
        diagnostic = dict(
            training_samples=len(trainRows),
            validation_samples=len(validationRows),
            split_seed=splitSeed,
            max_terms=termLimit,
            outputs=diagnostics,
        )
        return scores, normalized, diagnostic
