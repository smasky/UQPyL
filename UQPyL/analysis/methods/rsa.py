import numpy as np
import warnings
from typing import Optional
from scipy.stats import cramervonmises_2samp

from ..base import AnaIndex, AnalysisABC
from ...problem import ProblemABC as Problem


class RSA(AnalysisABC):
    """
    Regional Sensitivity Analysis (RSA)
    Sensitivity analysis based on regional output partitioning.

    Examples:
        >>> from UQPyL.doe import LHS
        >>> rsa_method = RSA(nRegion=20)
        >>> X = LHS('classic').sample(problem, 500)
        >>> Y = problem.evaluate(X, target="objs").objs
        >>> res = rsa_method.analyze(problem, X, Y, target="objs")

    References:
        [1] F. Pianosi et al., Sensitivity analysis of environmental models: A systematic review with practical workflow,
            Environmental Modelling & Software, vol. 79, pp. 214-232, May 2016,
            doi: 10.1016/j.envsoft.2016.02.008.
        [2] SALib, https://github.com/SALib/SALib
    """

    name = "RSA"

    def __init__(self, nRegion: int = 20, verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the RSA method for sensitivity analysis.

        Args:
            nRegion: Integer number of output regions, at least two.
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        super().__init__(verboseFlag, logFlag, saveFlag)
        self.set("nRegion", nRegion)

    def _analyzeCore(
        self,
        problem: Problem,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
        target: str = "objs",
        index: AnaIndex = "all",
    ) -> None:
        """
        Run RSA on the provided samples.

        Args:
            problem: Analysis problem.
            X: Input sample matrix.
            Y: Optional output matrix corresponding to `X`.
            meta: Optional sampling metadata for persistence only.
            target: Semantic label of `Y`, typically `objs` or `cons`.
            index: Output column selection.
        """

        # Set the problem instance for analysis

        nInput = problem.nInput
        nRegion = self.get("nRegion")
        if isinstance(nRegion, (bool, np.bool_)) or not isinstance(nRegion, (int, np.integer)) or nRegion < 2:
            raise ValueError("RSA nRegion must be an integer of at least 2.")
        if not np.all(np.isfinite(X)):
            raise ValueError("RSA requires finite input values.")

        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        calculationY = np.asarray(Y, dtype=float)
        if not np.all(np.isfinite(calculationY)):
            raise ValueError("RSA requires finite output values.")
        if len(Y) == 0:
            raise ValueError("RSA requires at least one input/output sample.")

        numY = Y.shape[1]

        S1 = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))

        row_label = self.outputLabels
        col_label_1 = problem.xLabels
        regionDiagnostics = []

        for i in range(numY):
            Y_i = calculationY[:, i : i + 1]

            # Define the sequence for dividing the input space into regions
            seq = np.linspace(0.0, 1.0, nRegion + 1)
            results = np.full((nRegion, nInput), np.nan)

            trr = Y_i.ravel()
            quants = self._outputQuantiles(trr, seq)
            regionSampleCounts = np.zeros(nRegion, dtype=int)
            validRegionCount = 0

            # Region eligibility depends on output membership, not the input axis.
            for binIndex in range(nRegion):
                lowerMask = quants[binIndex] <= trr if binIndex == 0 else quants[binIndex] < trr
                b = lowerMask & (trr <= quants[binIndex + 1])
                regionSampleCounts[binIndex] = np.count_nonzero(b)
                if self._has_samples(Y_i, b):
                    validRegionCount += 1
                    for d_i in range(nInput):
                        results[binIndex, d_i] = cramervonmises_2samp(X[b, d_i], X[~b, d_i]).statistic

            if np.all(trr == trr[0]):
                outputStatus = "constant_output"
            elif validRegionCount == 0:
                outputStatus = "insufficient_samples"
                warnings.warn(
                    f"RSA output '{row_label[i]}' has no valid region comparisons "
                    f"(n_samples={len(Y)}, nRegion={nRegion}); each region and its complement "
                    "need at least two samples. Increase the sample count or reduce nRegion. "
                    "S1 and S1_norm are zero placeholders, not evidence of insensitivity.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            else:
                outputStatus = "estimated"
            regionDiagnostics.append(
                dict(
                    output_label=row_label[i],
                    status=outputStatus,
                    valid_region_count=validRegionCount,
                    region_sample_counts=regionSampleCounts.tolist(),
                )
            )

            # Calculate the mean sensitivity index for each input factor
            validCounts = np.sum(~np.isnan(results), axis=0)
            sums = np.nansum(results, axis=0)
            results_star = np.divide(
                sums,
                validCounts,
                out=np.zeros_like(sums),
                where=validCounts > 0,
            )

            S1[i] = results_star
            total = np.sum(results_star)
            if np.isclose(total, 0.0):
                S1_norm[i] = 0.0
            else:
                S1_norm[i] = results_star / total

        res = [("S1", S1, row_label, col_label_1, "decsDim1"), ("S1_norm", S1_norm, row_label, col_label_1, "decsDim1")]

        self.state.extra["rsa_regions"] = dict(n_regions=int(nRegion), n_samples=len(Y), outputs=regionDiagnostics)
        self.recordResult(X, Y, res, target=target, meta=meta)

        return None

    @staticmethod
    def _outputQuantiles(values, probabilities):
        """Linear quantiles without overflowing a cross-zero subtraction."""
        ordered = np.sort(values)
        positions = (len(ordered) - 1) * probabilities
        lower = ordered[np.floor(positions).astype(int)]
        upper = ordered[np.ceil(positions).astype(int)]
        fractions = positions - np.floor(positions)
        sameSign = np.signbit(lower) == np.signbit(upper)
        quantiles = np.empty_like(lower)
        # Same-sign differences are finite. Match NumPy's interpolation from
        # the nearer endpoint without scaling away tiny distinct observations.
        differences = upper[sameSign] - lower[sameSign]
        weights = fractions[sameSign]
        quantiles[sameSign] = np.where(
            weights < 0.5,
            lower[sameSign] + differences * weights,
            upper[sameSign] - differences * (1 - weights),
        )
        # Opposite-sign weighted terms cannot overflow when added, unlike
        # upper-lower for endpoints near +/- the largest finite float.
        crossZero = ~sameSign
        quantiles[crossZero] = lower[crossZero] * (1 - fractions[crossZero]) + upper[crossZero] * fractions[crossZero]
        return quantiles

    def _has_samples(self, y, sel):
        """
        Check if the selected samples are sufficient for analysis.

        Each input-sample group needs at least two observations for the
        two-sample statistic. Constant output values within a group are valid.

        Args:
            y: Output values for one analyzed target.
            sel: Boolean mask of the selected region.

        Returns:
            Whether the region has enough samples for RSA.
        """
        return (np.count_nonzero(sel) >= 2) and (len(y[~sel]) >= 2)
