import numpy as np
import pytest

from UQPyL.surrogate.split import RandSelect, KFold
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.regression.linear_regression import LinearRegression
from UQPyL.surrogate.scaler import StandardScaler
from UQPyL.optimization.soea import DE


def assertGlobalStateEqual(before, after):
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize('splitter', [RandSelect(20), KFold(5)])
def test_split_is_reproducible_and_does_not_modify_input_or_global_state(splitter):
    X = np.arange(62).reshape(31, 2)
    saved = X.copy()
    globalState = np.random.get_state()
    first = splitter.split(X, seed=12)
    second = splitter.split(X, seed=12)
    third = splitter.split(X, seed=13)
    for a, b in zip(first, second):
        if isinstance(a, list):
            for aa, bb in zip(a, b): np.testing.assert_array_equal(aa, bb)
        else: np.testing.assert_array_equal(a, b)
    assert not np.array_equal(np.concatenate(first[1]) if isinstance(first[1], list) else first[1],
                              np.concatenate(third[1]) if isinstance(third[1], list) else third[1])
    np.testing.assert_array_equal(X, saved)
    assertGlobalStateEqual(globalState, np.random.get_state())


@pytest.mark.parametrize('n,k', [(11, 5), (7, 3), (5, 5), (10, 2)])
def test_kfold_every_sample_validated_once_and_balanced(n, k):
    X = np.zeros((n, 1))
    train, test = KFold(k).split(X, seed=8)
    np.testing.assert_array_equal(np.sort(np.concatenate(test)), np.arange(n))
    sizes = [len(t) for t in test]
    assert max(sizes)-min(sizes) <= 1
    for a, b in zip(train, test):
        assert set(a).isdisjoint(b)
        assert set(a) | set(b) == set(range(n))
    a, b = KFold(k).split(X, mode='single', seed=8)
    np.testing.assert_array_equal(a[0], train[0])
    np.testing.assert_array_equal(b[0], test[0])


@pytest.mark.parametrize('splitter', [RandSelect(20), KFold(5)])
def test_rng_objects_advance_and_conflicting_inputs_are_rejected(splitter):
    X = np.zeros((30, 1))
    rng = np.random.default_rng(2)
    before = repr(rng.bit_generator.state)
    splitter.split(X, rng=rng)
    assert before != repr(rng.bit_generator.state)
    with pytest.raises(ValueError, match='only one'):
        splitter.split(X, rng=rng, seed=1)


@pytest.mark.parametrize('factory', [lambda: KFold(1), lambda: KFold(2.5), lambda: RandSelect(0),
                                   lambda: RandSelect(100), lambda: RandSelect(np.nan)])
def test_invalid_split_parameters(factory):
    with pytest.raises(ValueError): factory()


def test_invalid_fold_count_and_mode():
    with pytest.raises(ValueError, match='exceed'): KFold(5).split(np.zeros((3, 1)))
    with pytest.raises(ValueError, match='mode'): KFold(2).split(np.zeros((3, 1)), mode='wrong')


@pytest.mark.parametrize('entry', ['gridTune', 'optTune'])
def test_real_tuning_reproducible_and_global_rng_unchanged(entry):
    X = np.linspace(0, 3, 30)[:, None]
    Y = X**2+.2*np.sin(X)
    def run():
        model = LinearRegression(scalers=(StandardScaler(), StandardScaler()), lossType='Ridge')
        optimizer = DE(nPop=6, maxFEs=12, verboseFlag=False, logFlag=False, saveFlag=False)
        tuner = AutoTuner(model, optimizer)
        kwargs = {'paraGrid': {'C': [-3., -1., 0.]}} if entry == 'gridTune' else {'paraList': ['C']}
        result = getattr(tuner, entry)(X, Y, ratio=20, tuneMode='joint', seed=42, **kwargs)
        return result, tuner.lastSplit, model.predict(X)
    globalState = np.random.get_state()
    first, split1, pred1 = run()
    second, split2, pred2 = run()
    for a, b in zip(first, second): np.testing.assert_allclose(a, b)
    for key in split1: np.testing.assert_array_equal(split1[key], split2[key])
    np.testing.assert_allclose(pred1, pred2)
    assertGlobalStateEqual(globalState, np.random.get_state())


def test_separate_mode_model_rng_is_seeded_and_replayable():
    class TrackingModel(LinearRegression):
        def __init__(self):
            super().__init__(lossType='Ridge')
            self.draws = []
        def fitHyper(self, X, Y):
            self.draws.append(self.rng.random())
            return super().fitHyper(X, Y)
    X = np.linspace(0, 2, 20)[:, None]
    tuner = AutoTuner(TrackingModel())
    kwargs = dict(paraGrid={'C': [-2, -1]}, ratio=20, seed=9, tuneMode='separate')
    first = tuner.gridTune(X, X**2, **kwargs)
    metadata = tuner.lastSplit.copy()
    second = tuner.gridTune(X, X**2, **kwargs)
    assert len(tuner.model.draws) == 6
    np.testing.assert_allclose(tuner.model.draws, np.repeat(tuner.model.draws[0], 6))
    for a, b in zip(first, second): np.testing.assert_allclose(a, b)
    for key in metadata: np.testing.assert_array_equal(metadata[key], tuner.lastSplit[key])
    with pytest.raises(ValueError, match='only one'):
        tuner.gridTune(X, X**2, rng=np.random.default_rng(1), **kwargs)


def test_randselect_uses_requested_percentage_without_fold_rounding():
    train, test = RandSelect(30).split(np.zeros((10, 1)), seed=3)
    assert len(test) == 3 and len(train) == 7
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(10))
