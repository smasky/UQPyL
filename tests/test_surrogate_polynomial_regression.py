import numpy as np

from UQPyL.surrogate.regression.polynomial_regression import PolynomialRegression


def test_polynomial_regression_fit_predict_smoke():
    # y = x^2
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    Y = (X**2)
    pr = PolynomialRegression(degree=2, onlyInteraction=False, lossType="Origin", fitIntercept=True)
    pr.fit(X, Y)
    pred = pr.predict(np.array([[4.0]]))
    assert pred.shape == (1, 1)
    assert np.isfinite(pred).all()


def test_polynomial_regression_default_tune_parameters_and_activation():
    model = PolynomialRegression(degree=2, lossType="Origin", onlyInteraction=False)

    assert model.getDefaultTuneParameters() == ["degree"]
    assert model.getDefaultTuneParameters(advanced=True) == ["degree", "lossType", "onlyInteraction"]
    assert not model.isParameterActive("C")

    model.setLossType("Ridge")
    assert model.isParameterActive("C")
    assert model.getDefaultTuneParameters(advanced=True) == ["degree", "lossType", "C", "onlyInteraction"]

    model.setLossType("Lasso")
    assert model.isParameterActive("C")
    assert model.isParameterActive("maxIter")
    assert model.isParameterActive("maxEpoch")
    assert model.isParameterActive("tol")
    assert model.isParameterActive("p0")


def test_polynomial_regression_default_degree_configuration_is_conservative():
    model = PolynomialRegression()

    assert model.degree == 2

    assert model.setting.getVals("degree") == 2
    assert model.setting.parLB["degree"].item() == 1
    assert model.setting.parUB["degree"].item() == 3
    assert model.setting.parType["degree"] == 1
    assert model.setting.parLog["degree"] is False


def test_polynomial_regression_only_interaction_changes_feature_count():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    full_model = PolynomialRegression(degree=2, onlyInteraction=False)
    interaction_model = PolynomialRegression(degree=2, onlyInteraction=True)

    full_features = full_model.polynomialFeatures(x)
    interaction_features = interaction_model.polynomialFeatures(x)

    assert full_features.shape[1] > interaction_features.shape[1]


def test_polynomial_regression_ridge_fit_predict_smoke():
    x = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y = 1.0 + 2.0 * x + 0.5 * x**2

    model = PolynomialRegression(degree=2, lossType="Ridge", fitIntercept=True, C=1e-6)
    model.fit(x, y)

    pred = model.predict(np.array([[0.5]]))
    assert pred.shape == (1, 1)
    assert np.isfinite(pred).all()


def test_polynomial_regression_apply_parameter_values_updates_model_context():
    model = PolynomialRegression()

    model.applyParameterValues(["degree", "lossType", "onlyInteraction"], [3, 1.2, 1.2])

    assert model.degree == 3
    assert model.lossType == "Ridge"
    assert model.onlyInteraction is True
    assert model.setting.getVals("degree") == 3
    assert model.setting.getVals("lossType") == "Ridge"
    assert model.setting.getVals("onlyInteraction") is True
    assert model.isParameterActive("C")

