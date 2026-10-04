"""MultiSurrogate 回归多输出的独立矩阵参照及单模型边界。"""

import numpy as np
import pytest

from UQPyL.surrogate.base import MultiSurrogate
from UQPyL.surrogate.regression import LinearRegression, PolynomialRegression
from UQPyL.surrogate.scaler import MinMaxScaler, StandardScaler


modelClasses = [LinearRegression, PolynomialRegression]


def newScaler(kind):
    if kind == "standard":
        return StandardScaler(muX=0.5, sitaX=2.0)
    if kind == "minmax":
        return MinMaxScaler(-2.0, 3.0)
    return None


def independentAffine(values, kind):
    if kind == "standard":
        spread = values.std(axis=0, ddof=1)
        scale = np.where(spread == 0, 1.0, spread) / 2.0
        offset = values.mean(axis=0) - 0.5 * scale
    elif kind == "minmax":
        spread = values.max(axis=0) - values.min(axis=0)
        scale = np.where(spread == 0, 1.0, spread) / 5.0
        offset = values.min(axis=0) + 2.0 * scale
    else:
        scale, offset = np.ones(values.shape[1]), np.zeros(values.shape[1])
    return offset, scale


def features(values, polynomial):
    if not polynomial:
        return values
    first, second = values[:, 0], values[:, 1]
    return np.column_stack((first, second, first**2, first * second, second**2))


def trainingCase(polynomial):
    first, second = np.meshgrid(np.linspace(-1.0, 1.0, 7), np.linspace(-0.7, 1.3, 5))
    inputs = np.column_stack((first.ravel(), second.ravel()))
    first, second = inputs[:, 0], inputs[:, 1]
    outputs = np.column_stack(
        (1 + 2 * first - 0.5 * second, -3 + 0.7 * first + 1.4 * second, np.full(len(inputs), 5.0))
    )
    if polynomial:
        outputs[:, 0] += 0.3 * first**2 + 0.2 * first * second
        outputs[:, 1] += 0.4 * second**2 - 0.1 * first * second
    probes = np.array([[-0.8, 0.2], [0.1, -0.4], [0.6, 0.9], [1.2, 1.5]])
    return inputs, outputs, probes


def independentPrediction(inputs, outputs, probes, polynomial, loss, kind, intercept):
    xOffset, xScale = independentAffine(inputs, kind)
    yOffset, yScale = independentAffine(outputs, kind)
    design = features((inputs - xOffset) / xScale, polynomial)
    testDesign = features((probes - xOffset) / xScale, polynomial)
    targets = (outputs - yOffset) / yScale
    if loss == "Origin":
        if intercept:
            design = np.column_stack((design, np.ones(len(design))))
            testDesign = np.column_stack((testDesign, np.ones(len(testDesign))))
        predictions = testDesign @ np.linalg.lstsq(design, targets, rcond=None)[0]
    else:
        designMean = design.mean(axis=0) if intercept else np.zeros(design.shape[1])
        targetMean = targets.mean(axis=0) if intercept else np.zeros(targets.shape[1])
        centered = design - designMean
        coefficient = np.linalg.solve(
            centered.T @ centered + 0.13 * np.eye(design.shape[1]), centered.T @ (targets - targetMean)
        )
        predictions = (testDesign - designMean) @ coefficient + targetMean
    return predictions * yScale + yOffset


@pytest.mark.parametrize("modelClass", modelClasses)
@pytest.mark.parametrize("loss", ["Origin", "Ridge"])
@pytest.mark.parametrize("kind", [None, "standard", "minmax"])
@pytest.mark.parametrize("intercept", [False, True])
def testContainerOutputsKeepAxesAndMatchIndependentMatrixSolution(modelClass, loss, kind, intercept):
    polynomial = modelClass is PolynomialRegression
    inputs, outputs, probes = trainingCase(polynomial)
    originalX, originalY = inputs.copy(), outputs.copy()
    inputs.flags.writeable = outputs.flags.writeable = False
    models = [
        modelClass(lossType=loss, C=0.13, fitIntercept=intercept, scalers=(newScaler(kind), newScaler(kind)))
        for _ in range(3)
    ]
    model = MultiSurrogate(3, models)
    model.fit(inputs, outputs)
    for count in [4, 1, 0]:
        expected = independentPrediction(inputs, outputs, probes[:count], polynomial, loss, kind, intercept)
        actual = model.predict(probes[:count])
        assert actual.shape == (count, 3)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(inputs, originalX)
    np.testing.assert_array_equal(outputs, originalY)


@pytest.mark.parametrize("modelClass", modelClasses)
@pytest.mark.parametrize("loss", ["Origin", "Ridge"])
def testPreparedSingleOutputModelsCombineWithCorrectOutputScaling(modelClass, loss):
    polynomial = modelClass is PolynomialRegression
    inputs, outputs, probes = trainingCase(polynomial)
    models = [
        modelClass(lossType=loss, C=0.13, scalers=(newScaler("standard"), newScaler("standard"))) for _ in range(3)
    ]
    model = MultiSurrogate(3, models)
    for index, child in enumerate(models):
        preparedX, preparedY = child.prepareTrainingData(inputs, outputs[:, index : index + 1])
        child.fitModel(preparedX, preparedY)
    actual = model.predict(probes)
    assert actual.shape == (4, 3)
    expected = independentPrediction(inputs, outputs, probes, polynomial, loss, "standard", True)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("modelClass", modelClasses)
@pytest.mark.parametrize("loss", ["Origin", "Ridge"])
def testRefitSingleModelsAndContainerAcceptSingleOutputVectors(modelClass, loss):
    polynomial = modelClass is PolynomialRegression
    inputs, outputs, probes = trainingCase(polynomial)
    models = [
        modelClass(lossType=loss, C=0.13, scalers=(newScaler("standard"), newScaler("standard"))) for _ in range(3)
    ]
    model = MultiSurrogate(3, models)
    for shift in [0, 2, -1, 0]:
        target = outputs + shift
        model.fit(inputs, target)
        models[0].fit(inputs, target[:, 0])
        actual = model.predict(probes[0])
        assert actual.shape == (1, 3)
        expected = independentPrediction(inputs, target, probes[:1], polynomial, loss, "standard", True)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("modelClass", modelClasses)
@pytest.mark.parametrize("entry", ["fit", "fitModel"])
def testLassoRejectsMultipleOutputsBeforeCallingNativeSolver(modelClass, entry, monkeypatch):
    inputs, outputs, probes = trainingCase(modelClass is PolynomialRegression)
    model = modelClass(lossType="Lasso", C=0.03, tolerance=1e-9)
    model.fit(inputs, outputs[:, :1])

    def unexpectedNativeCall(*args, **kwargs):
        raise AssertionError("Multiple outputs reached the single-output native solver")

    monkeypatch.setattr("UQPyL.surrogate.regression.lasso.compute_norms_X_col", unexpectedNativeCall)
    with pytest.raises(ValueError, match="single output.*MultiSurrogate"):
        getattr(model, entry)(inputs, outputs)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(probes)


@pytest.mark.parametrize("modelClass", modelClasses)
def testLassoMultiSurrogateStillMatchesIndependentSoftThresholdSolution(modelClass):
    inputs = np.linspace(-1.0, 1.0, 25).reshape(-1, 1)
    slopes, offsets = np.array([2.0, -0.8]), np.array([1.0, 3.0])
    outputs = inputs * slopes + offsets
    probes = np.array([[-0.7], [0.2], [0.8]])
    models = [modelClass(lossType="Lasso", C=0.03, tolerance=1e-9) for _ in slopes]
    container = MultiSurrogate(2, models)
    container.fit(inputs, outputs)
    actual = container.predict(probes)
    shrunk = np.sign(slopes) * np.maximum(np.abs(slopes) - 0.03 / np.mean(inputs**2), 0)
    assert actual.shape == (3, 2)
    np.testing.assert_allclose(actual, probes * shrunk + offsets, rtol=1e-9, atol=1e-9)
