"""Internal SUFI2 sampling must respect real mixed domains at every iteration."""

from copy import deepcopy

import numpy as np
import pytest

from UQPyL.calibration import SUFI2
from UQPyL.problem import ModelProblem


@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.parametrize("eliteSize", [1, 8])
def test_mixed_sampling_and_successive_elite_envelopes(seed, eliteSize):
    batches = []
    choices = [7.5, -2.25, 0.125, 3.5]

    def simulate(x):
        batches.append(x.copy())
        assert np.all(x[:, 1] == np.floor(x[:, 1]))
        assert np.all((x[:, 1] >= -1) & (x[:, 1] <= 2))
        assert np.isin(x[:, 2], choices).all()
        return np.stack([x[:, 0] + x[:, 1], x[:, 2]], axis=1)

    problem = ModelProblem(
        nInput=3,
        lb=[0, -1.8, 0],
        ub=[1, 2.8, 1],
        varType=[0, 1, 2],
        varSet={2: choices},
        simFunc=simulate,
        obs=np.array([0.5, 3.5]),
    )
    original = deepcopy(problem.varSet)
    result = SUFI2(maxIters=4, nSamples=24, explorationFraction=0, minRangeFraction=0).run(problem, eliteSize=eliteSize, seed=seed)
    assert len(batches) == 4
    active = choices.copy()
    for index, (batch, history) in enumerate(zip(batches, result.history.metricsHistory)):
        assert np.isin(batch[:, 2], active).all()
        scores = np.sqrt(np.mean((simulateValues(batch) - np.array([0.5, 3.5])) ** 2, axis=1))
        elite = batch[np.argsort(scores)[:eliteSize]]
        np.testing.assert_array_equal(history["updatedLb"], elite.min(0))
        np.testing.assert_array_equal(history["updatedUb"], elite.max(0))
        active = [value for value in active if elite[:, 2].min() <= value <= elite[:, 2].max()]
        assert history["updatedVarSet"][2] == active
        if index:
            prior = result.history.metricsHistory[index - 1]
            assert np.all(batch >= prior["updatedLb"])
            assert np.all(batch <= prior["updatedUb"])
    assert result.diagnostics["updatedVarSet"][2] == active
    assert problem.varSet == original
    np.testing.assert_array_equal(problem.lb, [[0, -1.8, 0]])


def simulateValues(x):
    return np.stack([x[:, 0] + x[:, 1], x[:, 2]], axis=1)


def test_discrete_envelope_retains_unobserved_legal_choices():
    problem = ModelProblem(
        nInput=1,
        lb=0,
        ub=1,
        varType=[2],
        varSet={0: [20.0, -5.0, 3.0, 10.0]},
        obs=np.array([1.0, 2.0]),
        simFunc=lambda x: np.stack([x[:, 0], 2 * x[:, 0]], axis=1),
    )
    result = SUFI2().run(problem, np.array([[-5.0], [10.0]]), eliteSize=2)
    assert result.diagnostics["updatedVarSet"][0] == [-5.0, 3.0, 10.0]


@pytest.mark.parametrize("variableType,values", [(1, [0.5]), (2, [0.5])])
def test_illegal_supplied_samples_fail_before_simulation(variableType, values):
    calls = []
    problem = ModelProblem(
        nInput=1,
        lb=0,
        ub=2,
        varType=[variableType],
        varSet={0: [0.25, 0.75]},
        obs=np.array([1.0, 2.0]),
        simFunc=lambda x: (calls.append(x)).reshape(len(x), -1),
    )
    with pytest.raises(ValueError):
        SUFI2().run(problem, np.array(values)[:, None], eliteSize=1)
    assert not calls
