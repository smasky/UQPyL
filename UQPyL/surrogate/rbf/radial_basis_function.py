from .._kernel import installKernel
import numpy as np
import warnings
from scipy.spatial.distance import cdist
from scipy.linalg import lu, solve_triangular, lstsq, null_space
from typing import Tuple, Optional, Literal

from .kernel import BaseKernel, Cubic
from ..base import SurrogateABC
from ..scaler import Scaler
from ..poly import PolyFeature


class RBF(SurrogateABC):
    """
    Radial basis function surrogate model.

    The model interpolates or smooths training data by combining radial basis
    responses with an optional polynomial tail, depending on the selected kernel.

    Examples:
        >>> model = RBF()
        >>> model.fit(xTrain, yTrain)
        >>> yPred = model.predict(xPred)

    References:
        [1] M. D. Buhmann, Radial Basis Functions: Theory and Implementations,
            Cambridge University Press, 2003.
    """

    name = "RBF"

    def __init__(
        self,
        scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
        polyFeature: PolyFeature = None,
        kernel: Optional[BaseKernel] = None,
        C_smooth: float = 0.0,
        C_smooth_attr: dict = {"ub": 1e5, "lb": 1e-5, "type": "float", "log": True},
    ):
        """
        Args:
            scalers: Tuple of input and output scalers.
            polyFeature: Polynomial features to be used.
            kernel: Kernel function for the RBF network.
            C_smooth: Finite, nonnegative smoothing strength applied only to
                the kernel-block diagonal with the kernel's sign.
            C_smooth_attr: Attribute for the smoothing parameter.
        """
        super().__init__(scalers, polyFeature)

        self.setting.set("C_smooth", C_smooth, C_smooth_attr)

        self.registerParameterApplier("kernel", self.setKernel)

        self.kernel = None
        self.setKernel(Cubic() if kernel is None else kernel)

    def _prepare_training_components(self, xTrain: np.ndarray):
        if hasattr(self.kernel, "initialize"):
            self.kernel.initialize(xTrain.shape[1])

    def setKernel(self, kernel: BaseKernel):
        return installKernel(self, kernel, BaseKernel)

    def setKernelChoices(self, kernels):
        self.registerChoiceParameter("kernel", [kernel.clone() for kernel in kernels], owner="kernel")
        return self

    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Fit the RBF model to the training data.

        Args:
            xTrain: Training input data.
            yTrain: Training output data.
        """
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)

        nSample, nFeature = xTrain.shape

        smooth = np.asarray(self.setting.get("C_smooth"), dtype=float)
        if smooth.size != 1 or not np.all(np.isfinite(smooth)) or np.any(smooth < 0):
            raise ValueError("C_smooth must be a finite, nonnegative scalar.")

        A_Matrix = self.kernel.get_A_Matrix(xTrain)
        # Preserve the polynomial tail and its zero constraint block.
        diagonal = np.arange(nSample)
        A_Matrix[diagonal, diagonal] += self.kernel.smoothingSign * smooth.item()

        P, L, U = lu(a=A_Matrix)
        degree = self.kernel.get_degree(nFeature)

        if degree:
            bias = np.vstack((yTrain, np.zeros((degree, yTrain.shape[1]))))
        else:
            bias = yTrain

        tail = A_Matrix[:nSample, nSample:]
        tailScale = np.max(np.abs(tail), axis=0) if degree else np.ones(0)
        tailScale = np.where(tailScale == 0, 1.0, tailScale)
        deficientTrend = bool(degree and np.linalg.matrix_rank(tail / tailScale) < degree)
        repeatedInputs = smooth.item() == 0 and len(np.unique(xTrain, axis=0)) < nSample

        # Solve the complete system: a pseudoinverse of each factor can
        # truncate polynomial constraints merely because input units changed.
        try:
            if deficientTrend or repeatedInputs:
                raise np.linalg.LinAlgError("Nonunique interpolation system.")
            solve = solve_triangular(U, solve_triangular(L, P.T @ bias, lower=True, unit_diagonal=True))
            solveInfo = {"method": "lu"}
        except np.linalg.LinAlgError:
            solve, solveInfo = self._fitSingularSystem(A_Matrix, yTrain, tail, tailScale, smooth.item())
            solveInfo.update(repeatedInputs=bool(repeatedInputs), deficientTrend=deficientTrend)
            warnings.warn(
                "RBF interpolation system is singular; using a constrained least-squares approximation. "
                f"Relative training residual={solveInfo['relativeTrainingResidual']:.3g}. "
                "Exact interpolation or unique extrapolation is not guaranteed; inspect fitState['linearSolve'].",
                RuntimeWarning,
                stacklevel=2,
            )

        if degree:
            coe_h = solve[nSample:, :]
        else:
            coe_h = 0

        self.fitState["coe_h"] = coe_h
        self.fitState["coe_lambda"] = solve[:nSample, :]
        self.fitState["linearSolve"] = solveInfo
        return self

    def _fitSingularSystem(self, matrix, targets, tail, tailScale, smooth):
        """Approximate training targets while retaining polynomial constraints."""
        nSample = len(targets)
        kernelBlock = matrix[:nSample, :nSample]
        basis = null_space((tail / tailScale).T) if tail.shape[1] else np.eye(nSample)
        projected = kernelBlock @ basis
        # Do not amplify roundoff-only columns when a null direction is also
        # annihilated by the kernel (for example, identical observations).
        tolerance = np.finfo(float).eps * nSample * np.max(np.abs(kernelBlock))
        if projected.shape[1]:
            projected[:, np.max(np.abs(projected), axis=0) <= tolerance] = 0.0
        design = np.hstack((projected, tail))
        columnScale = np.max(np.abs(design), axis=0)
        columnScale = np.where(columnScale == 0, 1.0, columnScale)
        coefficients, _, rank, _ = lstsq(design / columnScale, targets)
        coefficients = coefficients / columnScale[:, None]
        weights = basis @ coefficients[: basis.shape[1]]
        trend = coefficients[basis.shape[1] :]
        solution = np.vstack((weights, trend))
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError("RBF least-squares coefficients are not finite.")
        predicted = kernelBlock @ weights + tail @ trend - self.kernel.smoothingSign * smooth * weights
        if not np.all(np.isfinite(predicted)):
            raise np.linalg.LinAlgError("RBF least-squares predictions are not finite.")
        magnitude = max(float(np.max(np.abs(targets))), float(np.max(np.abs(predicted)))) or 1.0
        residual = (predicted / magnitude) - (targets / magnitude)
        targetNorm = np.linalg.norm(targets / magnitude)
        relativeResidual = np.linalg.norm(residual) / targetNorm if targetNorm else np.linalg.norm(residual)
        return solution, {
            "method": "constrained_lstsq",
            "rank": int(rank),
            "columns": design.shape[1],
            "relativeTrainingResidual": float(relativeResidual),
            "constraintMaxAbsResidual": float(np.max(np.abs(tail.T @ weights))) if tail.shape[1] else 0.0,
        }

    def predict(self, xPred: np.ndarray, returnStd: bool = False, returnVar: bool = False):
        """
        Predict outputs for given input data using the RBF model.

        Args:
            xPred: Input data for prediction.

        Returns:
            Predicted output data.
        """
        self._normalize_predict_flags(returnStd, returnVar)
        self.requireFitted("coe_h", "coe_lambda")

        xPred = self._transformX(xPred)
        _, nFeature = xPred.shape

        dist = cdist(xPred, self.xTrain)
        temp1 = np.dot(self.kernel.evaluate(dist), self.fitState["coe_lambda"])
        temp2 = np.zeros((temp1.shape[0], 1))

        degree = self.kernel.get_degree(nFeature)
        if degree:
            if degree > 1:
                temp2 = temp2 + np.dot(xPred, self.fitState["coe_h"][:-1, :])
            if degree > 0:
                temp2 = temp2 + np.repeat(self.fitState["coe_h"][-1:, :], temp1.shape[0], axis=0)

        return self._inverseTransformY(temp1 + temp2)
