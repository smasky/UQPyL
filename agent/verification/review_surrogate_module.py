"""Inspect surrogate numerical and public-input boundaries without changing models.

Run in conda py312 with PYTHONPATH=. and one BLAS/OMP thread. This is an
audit: confirmed discrepancies are recorded, rather than treated as passing
regression tests. SciPy/NumPy references retain independent equations.
"""

from collections import Counter
from itertools import combinations, combinations_with_replacement
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import CubicSpline, RBFInterpolator
from scipy.linalg import lu, solve_triangular

from UQPyL.optimization.soea import GA
from UQPyL.surrogate import AutoTuner, MinMaxScaler, MultiSurrogate, PolyFeature, StandardScaler
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as GpRbf
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Exp
from UQPyL.surrogate.metric import nse, r_square
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.rbf.kernel import Cubic
from UQPyL.surrogate.regression import LinearRegression, PolynomialRegression
from UQPyL.surrogate.svr import SVR


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def capture(action):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = {"value": action(), "accepted": True}
        except Exception as error:
            result = {"accepted": False, "error_type": type(error).__name__, "message": str(error)}
    result["warnings"] = [str(item.message) for item in caught]
    return result


records = []


def record(group, case, actual, expected, correct, **details):
    records.append(
        dict(
            group=group,
            case=case,
            status="control" if correct else "issue",
            actual=actual,
            expected=expected,
            **details,
        )
    )


def close(actual, expected, tolerance=1e-7):
    return bool(np.all(np.isfinite(actual)) and np.allclose(actual, expected, rtol=tolerance, atol=tolerance))


# SR01: cubic interpolation and its linear tail must survive physical units.
baseX = np.array([0.0, 0.1, 0.27, 0.46, 0.65, 0.82, 1.0])[:, None]
baseQuery = np.array([0.07, 0.21, 0.53, 0.91])[:, None]
for kind, targets in [("linear", 2 + 3 * baseX), ("nonlinear", 2 + 3 * baseX + np.sin(5 * baseX))]:
    splineReference = CubicSpline(baseX.ravel(), targets.ravel(), bc_type="natural")(baseQuery.ravel())[:, None]
    for factor in [1.0, 100.0, 1000.0, 10000.0]:
        for scaled in [False, True]:
            model = RBF(kernel=Cubic(), scalers=(MinMaxScaler() if scaled else None, None)).fit(baseX * factor, targets)
            prediction = model.predict(baseQuery * factor)
            scipyReference = RBFInterpolator(baseX * factor, targets, kernel="cubic")(baseQuery * factor)
            assert close(scipyReference, splineReference, 1e-11)
            # A controlled replacement solves the same LU factors without a
            # pseudoinverse cutoff; the production model remains unchanged.
            matrix = model.kernel.get_A_Matrix(model.xTrain)
            permutation, lower, upper = lu(matrix)
            right = np.vstack([targets, np.zeros((2, 1))])
            coefficients = solve_triangular(
                upper, solve_triangular(lower, permutation.T @ right, lower=True, unit_diagonal=True)
            )
            preparedQuery = model._transformX(baseQuery * factor)
            triangularPrediction = (
                np.abs(preparedQuery - model.xTrain.T) ** 3 @ coefficients[: len(baseX)]
                + np.column_stack([preparedQuery, np.ones(len(preparedQuery))]) @ coefficients[len(baseX) :]
            )
            assert close(triangularPrediction, splineReference, 1e-11)
            singularValues = np.linalg.svd(upper, compute_uv=False)
            cutoff = np.max(singularValues) * max(upper.shape) * np.finfo(float).eps
            record(
                "SR01",
                f"{kind}:range={factor}:xScaler={scaled}",
                prediction,
                splineReference,
                close(prediction, splineReference),
                max_abs_diff=float(np.max(np.abs(prediction - splineReference))),
                training_max_abs_diff=float(np.max(np.abs(model.predict(baseX * factor) - targets))),
                triangular_max_abs_diff=float(np.max(np.abs(triangularPrediction - splineReference))),
                upper_pseudoinverse_rank=int(np.count_nonzero(singularValues > cutoff)),
                full_matrix_size=len(upper),
            )

# SR02: integer numerical inputs should agree with equivalent floating data.
integerX = np.arange(8)[:, None]
integerY = 2 * integerX + 1
for modelClass in [LinearRegression, PolynomialRegression]:
    for intercept in [False, True]:
        floatModel = modelClass(lossType="Lasso", fitIntercept=intercept).fit(
            integerX.astype(float), integerY.astype(float)
        )
        reference = floatModel.predict(integerX.astype(float))
        for xType, yType in [("float64", "float64"), ("int64", "float64"), ("float64", "int64"), ("int64", "int64")]:

            def fitInteger():
                fitted = modelClass(lossType="Lasso", fitIntercept=intercept).fit(
                    integerX.astype(xType), integerY.astype(yType)
                )
                return fitted.predict(integerX.astype(float))

            actual = capture(fitInteger)
            record(
                "SR02",
                f"{modelClass.__name__}:intercept={intercept}:X={xType}:Y={yType}",
                actual,
                reference,
                actual["accepted"] and close(actual["value"], reference),
            )

# SR03: nu-SVR default search bounds include invalid native parameters.
svrX = np.linspace(-1, 1, 30)[:, None]
svrY = svrX**2 + svrX
for seed in [1, 2, 3]:
    for corrected in [False, True]:
        options = {"nu_attr": {"ub": 1.0, "lb": 1e-5, "type": "float", "log": True}} if corrected else {}
        model = SVR(symbol="nu-SVR", C=1.0, **options)
        optimizer = GA(nPop=8, maxFEs=8, verboseFlag=False, saveFlag=False, logFlag=False)
        tuner = AutoTuner(model, optimizer)
        result = capture(lambda: tuner.optTune(svrX, svrY, paraList=["nu"], seed=seed, ratio=30, tuneMode="joint"))
        record(
            "SR03",
            f"seed={seed}:corrected_bounds={corrected}",
            result,
            "finite completed search with 0 < nu <= 1",
            result["accepted"] and bool(np.all(np.isfinite(result["value"][1]))),
            candidates=len(tuner.lastReport["candidates"]),
            report=tuner.getReport(),
        )

# SR04: dimensionless scores and tuning eligibility are invariant to output units.
metricY = np.arange(1.0, 5.0)[:, None]
metricPrediction = metricY + np.array([-0.2, 0.2, -0.2, 0.2])[:, None]
for factor in [1.0, 1e-100, 1e-200, 1e100, 1e160]:
    for metric in [r_square, nse]:
        actual = capture(lambda: float(metric(metricY * factor, metricPrediction * factor)))
        record(
            "SR04",
            f"{metric.__name__}:scale={factor}",
            actual,
            0.968,
            actual["accepted"] and close(actual["value"], 0.968),
        )
    tunerX = np.linspace(-1, 1, 16)[:, None]
    tunerY = (tunerX**2 + 2 * tunerX + 3) * factor
    tuner = AutoTuner(PolynomialRegression(scalers=(None, MinMaxScaler())))
    actual = capture(
        lambda: tuner.gridTune(
            tunerX,
            tunerY,
            paraGrid={"degree": [2]},
            tuneMode="joint",
            splitIndices=(np.arange(12), np.arange(12, 16)),
        )
    )
    record(
        "SR04",
        f"AutoTuner:scale={factor}",
        actual,
        "degree 2, R2=1",
        actual["accepted"] and close(actual["value"][1], 1.0),
    )

# SR05: known standard deviation computed in ordinary units before conversion.
normalStd = np.sqrt(5.0 / 3.0)
standardReference = (metricY - 2.5) / normalStd
for factor in [1.0, 1e-100, 1e-200, 1e100, 1e160]:
    scaler = StandardScaler()
    actual = capture(lambda: scaler.fit_transform(metricY * factor))
    record(
        "SR05",
        f"StandardScaler:scale={factor}",
        actual,
        standardReference,
        actual["accepted"] and close(actual["value"], standardReference),
        fitted_std=getattr(scaler, "sita", None),
        expected_std=normalStd * factor,
    )

# SR06: independently calculate single-output GP/Kriging posterior moments.
gpX = np.linspace(0, 1, 9)[:, None]
gpY = 2 + np.sin(3 * gpX)
gpQuery = np.array([[0.17], [1.4]])
unitOffset = np.min(gpY)
unitSpan = np.ptp(gpY)
unitY = (gpY - unitOffset) / unitSpan
for family in ["GPR", "KRG"]:
    if family == "GPR":
        covariance = np.exp(-0.5 * ((gpX - gpX.T) / 0.3) ** 2) + 0.01 * np.eye(len(gpX))
        cross = np.exp(-0.5 * ((gpQuery - gpX.T) / 0.3) ** 2)
        unitMean = cross @ np.linalg.solve(covariance, unitY)
        unitVar = 1 - np.einsum("ij,ji->i", cross, np.linalg.solve(covariance, cross.T))
    else:
        covariance = np.exp(-2 * np.abs(gpX - gpX.T)) + (10 + len(gpX)) * np.spacing(1) * np.eye(len(gpX))
        cross = np.exp(-2 * np.abs(gpQuery - gpX.T))
        ones = np.ones_like(unitY)
        invOnes = np.linalg.solve(covariance, ones)
        trend = float((ones.T @ np.linalg.solve(covariance, unitY) / (ones.T @ invOnes)).item())
        residual = unitY - trend
        invResidual = np.linalg.solve(covariance, residual)
        unitMean = trend + cross @ invResidual
        processVar = float((residual.T @ invResidual / len(gpX)).item())
        unitVar = processVar * (
            1
            - np.einsum("ij,ji->i", cross, np.linalg.solve(covariance, cross.T))
            + ((cross @ invOnes).ravel() - 1) ** 2 / float((ones.T @ invOnes).item())
        )
    referenceMean = unitMean * unitSpan + unitOffset
    referenceStd = np.sqrt(np.maximum(unitVar, 0))[:, None] * unitSpan

    def newModel():
        if family == "GPR":
            return GPR(
                kernel=GpRbf(length_scale=0.3, length_attr=None), C=0.01, C_attr=None, scalers=(None, MinMaxScaler())
            )
        return KRG(kernel=Exp(theta=2.0, theta_attr=None), scalers=(None, MinMaxScaler()))

    for factor in [1.0, 1e-100, 1e-200, 1e100, 1e160]:
        model = newModel().fit(gpX, gpY * factor)
        actual = capture(lambda: model.predict(gpQuery, returnStd=True))
        record(
            "SR06",
            f"{family}:scale={factor}",
            actual,
            {"mean": referenceMean * factor, "std": referenceStd * factor},
            actual["accepted"]
            and close(actual["value"][0] / factor, referenceMean)
            and close(actual["value"][1] / factor, referenceStd),
        )
    mixed = MultiSurrogate(2, [newModel(), newModel()]).fit(gpX, np.column_stack([gpY[:, 0], gpY[:, 0] * 1e-200]))
    actual = capture(lambda: mixed.predict(gpQuery, returnStd=True))
    record(
        "SR06",
        f"{family}:MultiSurrogate:mixed_units",
        actual,
        {
            "mean": np.hstack([referenceMean, referenceMean * 1e-200]),
            "std": np.hstack([referenceStd, referenceStd * 1e-200]),
        },
        actual["accepted"]
        and close(actual["value"][1] / np.array([1.0, 1e-200]), np.hstack([referenceStd, referenceStd])),
    )

# SR07: malformed data and nonfinite/negative noise must not create a fitted GP.
validationX = np.linspace(0, 1, 7)[:, None]
validationY = np.sin(3 * validationX)
for kind in ["positive_noise", "negative_noise", "nan_noise", "nan_targets", "nan_targets_scaled"]:
    noise = -0.01 if kind == "negative_noise" else np.nan if kind == "nan_noise" else 0.01
    targets = validationY.copy()
    if kind.startswith("nan_targets"):
        targets[3] = np.nan
    scaler = MinMaxScaler() if kind.endswith("scaled") else None
    model = GPR(kernel=GpRbf(length_scale=0.05, length_attr=None), C=noise, C_attr=None, scalers=(None, scaler))
    fitResult = capture(lambda: model.fit(validationX, targets) is model)
    details = {"fitted_state_keys": sorted(model.fitState)}
    if fitResult["accepted"]:
        details["mean_prediction"] = capture(lambda: model.predict(validationX))
        details["variance_prediction"] = capture(lambda: model.predict(validationX, returnVar=True))
    expectedAccepted = kind == "positive_noise"
    record(
        "SR07",
        kind,
        fitResult,
        "successful finite fit" if expectedAccepted else "reject before fitting",
        fitResult["accepted"] == expectedAccepted,
        **details,
    )

# Normal-scale control: feature expansion agrees with independent monomial lists.
for nFeatures in range(1, 6):
    data = np.arange(1.0, 3 * nFeatures + 1).reshape(3, nFeatures)
    for degree in range(1, 8):
        for interaction in [False, True]:
            for bias in [False, True]:
                combine = combinations if interaction else combinations_with_replacement
                terms = [
                    np.prod(data[:, indices], axis=1)
                    for order in range(1, degree + 1)
                    for indices in combine(range(nFeatures), order)
                ]
                expected = np.column_stack(([np.ones(len(data))] if bias else []) + terms)
                actual = capture(lambda: PolyFeature(degree, bias, interaction).transform(data))
                correct = actual["accepted"] and np.array_equal(actual["value"], expected)
                record(
                    "control_poly",
                    f"n={nFeatures}:degree={degree}:interaction={interaction}:bias={bias}",
                    {
                        "accepted": actual["accepted"],
                        "shape": np.shape(actual.get("value")),
                        "warnings": actual["warnings"],
                    },
                    {"shape": expected.shape},
                    correct,
                )

summary = {
    "total": len(records),
    "statuses": dict(Counter(item["status"] for item in records)),
    "issues_by_group": dict(Counter(item["group"] for item in records if item["status"] == "issue")),
}
payload = {"environment": "conda py312; single BLAS/OMP thread", "summary": summary, "records": records}
destination = Path(__file__).with_name("1002-surrogate-module-after-fixes.json")
destination.write_text(json.dumps(serializable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Evidence: {destination}")
