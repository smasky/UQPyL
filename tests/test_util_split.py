import numpy as np

from UQPyL.util.split import RandSelect, KFold


def test_randselect_split_small_dataset_no_div0():
    X = np.arange(10).reshape(-1, 1)
    train, test = RandSelect(pTest=2).split(X)  # 2% used to crash
    assert len(test) >= 1
    assert len(train) >= 1


def test_randselect_split_single_sample():
    X = np.array([[1.0]])
    train, test = RandSelect(pTest=50).split(X)
    assert train.shape == (1,)
    assert test.shape == (0,)


def test_kfold_split_modes_shapes():
    X = np.arange(20).reshape(-1, 1)
    kf = KFold(n_splits=5)
    trains, tests = kf.split(X, mode="full")
    assert len(trains) == 5 and len(tests) == 5
    train1, test1 = kf.split(X, mode="single")
    assert len(train1) == 1 and len(test1) == 1

