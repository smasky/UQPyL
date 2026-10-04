"""Explicitly flattened simulations retain the observation mask's C-order meaning."""

import numpy as np
import pytest

from UQPyL.problem import ModelProblem
from UQPyL.calibration import ES, IES, GLUE, SUFI2


@pytest.mark.parametrize("explicitGrid", [False, True])
@pytest.mark.parametrize("entry", ["evaluate", "simulate", "flattenSim"])
def testMaskedNanMatchesExplicitUserFlattening(explicitGrid, entry):
    mask = np.array([[False, True], [False, False]])
    sims = np.array([[[1.0, np.nan], [3.0, 4.0]], [[2.0, np.nan], [5.0, 6.0]]])
    values = sims.reshape(2, 4) if explicitGrid else np.array([[1., np.nan, 3., 4.], [2., np.nan, 5., 6.]])
    problem = ModelProblem(nInput=1, lb=0, ub=1, obs=np.ones(4), mask=mask.reshape(-1), simFunc=lambda x: values.copy())
    if entry == "flattenSim":
        result = problem.flattenSim(values)
    else:
        result = getattr(problem, entry)([[0.0], [1.0]]).sims
    np.testing.assert_array_equal(result.reshape(2, 4), sims.reshape(2, 4))


@pytest.mark.parametrize("shape", [(2, 4), (2, 2, 2), (2, 1, 4), (2, 3)])
def testUnmaskedOrUnalignedNanRemainsInvalid(shape):
    sims = np.ones(shape)
    sims.reshape(2, -1)[:, 0] = np.nan
    problem = ModelProblem(
        nInput=1,
        lb=0,
        ub=1,
        obs=np.ones(4),
        mask=np.array([False, True, False, False]),
        simFunc=lambda x: sims,
    )
    with pytest.raises(ValueError, match="NaN" if shape == (2, 4) else "2D|column count"):
        problem.evaluate([[0.0], [1.0]])


@pytest.mark.parametrize("methodClass", [ES, IES, GLUE, SUFI2])
def testCalibrationMaskedNanIsInvariantToExplicitFlattening(methodClass):
    samples = np.linspace(-1.0, 1.0, 24)[:, None]
    results = []
    for flat in (False, True):

        def simulate(x):
            values = x * np.array([[1.0, 2.0, 3.0, 4.0]])
            values[:, 1] = np.nan
            return values if flat else values.reshape(len(x), 2, 2).reshape(len(x), 4)

        problem = ModelProblem(
            nInput=1,
            lb=-5,
            ub=5,
            obs=np.array([0.2, np.nan, 0.6, 0.8]),
            mask=np.array([False, True, False, False]),
            simFunc=simulate,
        )
        method = methodClass(maxIters=2, seed=17) if methodClass is IES else methodClass()
        options = (
            {"r": np.eye(3)}
            if methodClass in (ES, IES)
            else ({"threshold": 100} if methodClass is GLUE else {"eliteSize": 8, "seed": 17})
        )
        results.append(method.run(problem, samples, **options))
    for field in ("samples", "simulations", "scores"):
        np.testing.assert_array_equal(getattr(results[0], field), getattr(results[1], field))


@pytest.mark.parametrize("scaled", [False, True])
def testSurrogatePredictionReportsAmbiguousVectorBeforeLinearAlgebra(scaled):
    from UQPyL.surrogate.regression import LinearRegression
    from UQPyL.surrogate.scaler import StandardScaler

    model = LinearRegression(scalers=(StandardScaler(), None) if scaled else (None, None))
    x = np.arange(5.0)
    model.fit(x, 2 * x)
    with pytest.raises(ValueError, match=r"one sample.*reshape"):
        model.predict(x)
    np.testing.assert_allclose(model.predict(x[:, None]), (2 * x)[:, None], atol=1e-12)
    np.testing.assert_allclose(model.predict(np.array([2.0])), [[4.0]], atol=1e-12)
