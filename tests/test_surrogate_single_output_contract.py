"""单模型单输出、预处理/调参边界和 MultiSurrogate 的统一职责。"""

from types import SimpleNamespace

import numpy as np
import pytest

from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as GpRbf
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Guass
from UQPyL.surrogate.mars import MARS
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.regression import LinearRegression, PolynomialRegression
from UQPyL.surrogate.scaler import StandardScaler
from UQPyL.surrogate.svr import SVR


modelNames = [
    "RBF",
    "GPR",
    "KRG",
    "MARS",
    "SVR",
    "LR_Origin",
    "LR_Ridge",
    "LR_Lasso",
    "PR_Origin",
    "PR_Ridge",
    "PR_Lasso",
]


def newModel(name):
    scalers = (StandardScaler(), StandardScaler())
    if name == "GPR":
        return GPR(kernel=GpRbf(length_attr=None), C_attr=None, scalers=scalers)
    if name == "KRG":
        return KRG(kernel=Guass(theta_attr=None), scalers=scalers)
    if name.startswith(("LR_", "PR_")):
        modelClass = LinearRegression if name.startswith("LR_") else PolynomialRegression
        return modelClass(lossType=name.split("_")[1], scalers=scalers)
    return {"RBF": RBF, "MARS": MARS, "SVR": SVR}[name](scalers=scalers)


def forbidden(*args, **kwargs):
    raise AssertionError("Invalid multi-output data reached preprocessing, fitting or search")


def blockBackend(model, entry, monkeypatch):
    if entry in {"fit", "prepareTrainingData"}:
        monkeypatch.setattr(model.yScaler, "fit", forbidden)
    elif entry == "fitHyper" and isinstance(model, (GPR, KRG)):
        monkeypatch.setattr(model, "_optimizeHyper", forbidden)
    elif isinstance(model, GPR):
        monkeypatch.setattr(model, "_objfunc", forbidden)
    elif isinstance(model, KRG):
        monkeypatch.setattr(model, "_initialize", forbidden)
    elif isinstance(model, RBF):
        monkeypatch.setattr(model.kernel, "get_A_Matrix", forbidden)
    elif isinstance(model, MARS):
        monkeypatch.setattr(model, "_scrape_labels", forbidden)
    elif isinstance(model, SVR):
        monkeypatch.setattr(model, "_build_parameter", forbidden)
    elif model.lossType == "Lasso":
        monkeypatch.setattr("UQPyL.surrogate.regression.lasso.compute_norms_X_col", forbidden)
    else:
        monkeypatch.setattr(model, "fit" + model.lossType, forbidden)


@pytest.mark.parametrize("name", modelNames)
@pytest.mark.parametrize("entry", ["fit", "prepareTrainingData", "fitModel", "fitHyper"])
def testAllModelTrainingEntriesRejectMultipleOutputsBeforeWork(name, entry, monkeypatch):
    inputs = np.linspace(-1, 1, 25)[:, None]
    outputs = np.column_stack((2 * inputs[:, 0] + 1, -inputs[:, 0] + 3))
    model = newModel(name)
    blockBackend(model, entry, monkeypatch)
    with pytest.raises(ValueError, match="single output.*MultiSurrogate"):
        getattr(model, entry)(inputs, outputs)
    assert model.fitState == {}
    assert not model.xScaler.fitted and not model.yScaler.fitted


@pytest.mark.parametrize("name", modelNames)
def testSingleOutputVectorAndMatrixKeepSamePredictions(name):
    inputs = np.linspace(-1, 1, 25)[:, None]
    values = 2 * inputs[:, 0] + 1
    probes = np.array([[-0.7], [0.2], [0.8]])
    predictions = []
    for target in [values, values[:, None]]:
        model = newModel(name).fit(inputs, target)
        prediction = model.predict(probes)
        assert prediction.shape == (3, 1) and model.yTrain.shape == (25, 1)
        assert np.all(np.isfinite(prediction))
        predictions.append(prediction)
    # This dense Gaussian KRG fixture has a correlation condition number ~1e15.
    # Check both layout equivalence and accuracy against the independent truth.
    tolerance = 1e-8 if name == "KRG" else 1e-12
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=tolerance, atol=tolerance)
    if name == "KRG":
        for prediction in predictions:
            np.testing.assert_allclose(prediction, 2 * probes + 1, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", ["LR_Origin", "GPR", "KRG", "MARS", "SVR"])
@pytest.mark.parametrize("entry", ["fit", "fitModel", "fitHyper"])
def testRejectedRefitDoesNotLeavePreviousPredictionAvailable(name, entry):
    inputs = np.linspace(-1, 1, 25)[:, None]
    outputs = np.column_stack((2 * inputs[:, 0] + 1, -inputs[:, 0] + 3))
    model = newModel(name).fit(inputs, outputs[:, :1])
    with pytest.raises(ValueError, match="single output.*MultiSurrogate"):
        getattr(model, entry)(inputs, outputs)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(inputs[:2])


@pytest.mark.parametrize("entry", ["gridTune", "optTune"])
@pytest.mark.parametrize("tuneMode", ["joint", "separate"])
def testTunerRejectsMultipleOutputsBeforeSplittingOrSearching(entry, tuneMode, monkeypatch):
    inputs = np.linspace(-1, 1, 25)[:, None]
    outputs = np.column_stack((2 * inputs[:, 0] + 1, -inputs[:, 0] + 3))
    model = newModel("LR_Ridge").fit(inputs, outputs[:, :1])
    tuner = AutoTuner(model, SimpleNamespace(run=forbidden))
    monkeypatch.setattr(tuner, "_splitData", forbidden)
    arguments = {"paraGrid": {"C": [np.log(0.1)]}} if entry == "gridTune" else {"paraList": ["C"]}
    with pytest.raises(ValueError, match="single output.*MultiSurrogate"):
        getattr(tuner, entry)(inputs, outputs, ratio=20, tuneMode=tuneMode, seed=7, **arguments)
    assert tuner.lastReport["status"] == "failed" and tuner.lastReport["fit_calls"] == 0
    assert tuner.lastSplit is None and model.fitState == {}


@pytest.mark.parametrize("shape", [(25, 0), (25, 3), (25, 2, 1), (24, 2), ()])
def testMultiSurrogateValidatesWholeTargetBeforeFittingChildren(shape, monkeypatch):
    models = [newModel("LR_Origin"), newModel("LR_Ridge")]
    container = MultiSurrogate(2, models)
    for model in models:
        monkeypatch.setattr(model, "fit", forbidden)
    with pytest.raises(ValueError, match="outputs|shape|sample"):
        container.fit(np.zeros((25, 1)), np.ones(shape))


@pytest.mark.parametrize("distribution", ["std", "var"])
def testMultiSurrogateGaussianUncertaintyMatchesIndependentMatrixSolutions(distribution):
    inputs = np.array([[0.0], [0.2], [0.55], [0.85], [1.0]])
    outputs = np.column_stack((np.sin(4 * inputs[:, 0]), 20 * np.cos(3 * inputs[:, 0]) + 50))
    probes = np.array([[0.13], [0.4], [0.72]])
    lengths, noises = [0.5, 0.9], [0.02, 0.07]
    models = [
        GPR(kernel=GpRbf(length_scale=length, length_attr=None), C=noise, C_attr=None, scalers=(None, StandardScaler()))
        for length, noise in zip(lengths, noises)
    ]
    container = MultiSurrogate(2, models)
    assert container.fit(inputs, outputs) is container
    assert container.supportsUncertainty
    kwargs = {"returnStd": True} if distribution == "std" else {"returnVar": True}
    actualMean, actualUncertainty = container.predict(probes, **kwargs)
    assert actualMean.shape == actualUncertainty.shape == (3, 2)
    for index, (length, noise) in enumerate(zip(lengths, noises)):
        covariance = np.exp(-0.5 * (inputs - inputs.T) ** 2 / length**2) + noise * np.eye(len(inputs))
        cross = np.exp(-0.5 * (probes - inputs.T) ** 2 / length**2)
        mean, spread = outputs[:, index].mean(), outputs[:, index].std(ddof=1)
        expectedMean = cross @ np.linalg.solve(covariance, (outputs[:, index] - mean) / spread) * spread + mean
        variance = (1 - np.sum(cross * np.linalg.solve(covariance, cross.T).T, axis=1)) * spread**2
        expectedUncertainty = np.sqrt(variance) if distribution == "std" else variance
        np.testing.assert_allclose(actualMean[:, index], expectedMean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actualUncertainty[:, index], expectedUncertainty, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("flags", [{"returnStd": True}, {"returnVar": True}, {"returnStd": True, "returnVar": True}])
def testMultiSurrogateUncertaintyRequestsValidateBeforePrediction(flags, monkeypatch):
    models = [newModel("LR_Origin"), newModel("GPR")]
    container = MultiSurrogate(2, models)
    assert not container.supportsUncertainty
    for model in models:
        monkeypatch.setattr(model, "predict", forbidden)
    error = ValueError if len(flags) == 2 else NotImplementedError
    with pytest.raises(error):
        container.predict(np.zeros((3, 1)), **flags)


@pytest.mark.parametrize("badShape", [(3, 2), (2, 1)])
def testMultiSurrogateRejectsChildPredictionWithWrongOutputOrSampleAxis(badShape, monkeypatch):
    models = [newModel("LR_Origin"), newModel("LR_Ridge")]
    container = MultiSurrogate(2, models)
    inputs = np.linspace(-1, 1, 25)[:, None]
    container.fit(inputs, np.column_stack((2 * inputs[:, 0] + 1, -inputs[:, 0] + 3)))
    monkeypatch.setattr(models[0], "predict", lambda values: np.zeros(badShape))
    with pytest.raises(ValueError, match="prediction.*shape"):
        container.predict(inputs[:3])


@pytest.mark.parametrize("failure", ["shape", "child"])
def testFailedContainerRefitInvalidatesAllChildrenAndCanRecover(failure, monkeypatch):
    inputs = np.linspace(-1, 1, 25)[:, None]
    outputs = np.column_stack((2 * inputs[:, 0] + 1, -inputs[:, 0] + 3))
    models = [newModel("LR_Origin"), newModel("LR_Ridge")]
    container = MultiSurrogate(2, models).fit(inputs, outputs)

    def failChild(*args):
        raise ValueError("injected child failure")

    with monkeypatch.context() as patch:
        if failure == "child":
            patch.setattr(models[1], "fit", failChild)
        with pytest.raises(ValueError):
            container.fit(inputs, outputs[:-1] if failure == "shape" else outputs)
    assert all(model.fitState == {} and model.xTrain is None and model.yTrain is None for model in models)
    with pytest.raises(RuntimeError, match="fitted"):
        container.predict(inputs[:3])
    assert container.fit(inputs, outputs) is container
    assert container.predict(inputs[:3]).shape == (3, 2)


def testMultiSurrogateDerivativeRequiresSupportingModels():
    container = MultiSurrogate(2, [newModel("LR_Origin"), newModel("MARS")])
    with pytest.raises(NotImplementedError, match="predict_deriv"):
        container.predict_deriv(np.zeros((3, 1)))
