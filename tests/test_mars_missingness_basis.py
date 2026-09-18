"""Exercise Python wrappers and C-level dispatch for missingness bases."""

import pickle

import numpy as np
import pytest

from UQPyL.surrogate.mars.core._basis import (
    Basis, ConstantBasisFunction, LinearBasisFunction, MissingnessBasisFunction,
)


def makeCase(complement):
    values = np.array([[1., -2.], [2., 3.], [3., 4.], [4., -5.]])
    missing = np.array([[0, 0], [1, 0], [1, 0], [0, 0]], dtype=np.uint8)
    root = ConstantBasisFunction()
    parent = LinearBasisFunction(root, 1)
    child = MissingnessBasisFunction(parent, 0, complement)
    mask = (1 - missing[:, 0]) if complement else missing[:, 0]
    return values, missing, root, parent, child, mask


@pytest.mark.parametrize('complement', [False, True])
@pytest.mark.parametrize('mode', ['default', 'positional', 'keyword', 'nonrecursive'])
def testMissingnessApplyWrapper(complement, mode):
    values, missing, _, _, child, mask = makeCase(complement)
    # A strided output also exercises the ndarray wrapper used by Basis.transform.
    output = np.full((4, 2), 7.)
    column = output[:, 0]
    if mode == 'default':
        child.apply(values, missing, column)
    elif mode == 'positional':
        child.apply(values, missing, column, True)
    elif mode == 'keyword':
        child.apply(values, missing, column, recurse=True)
    else:
        child.apply(values, missing, column, recurse=False)
    expected = (7. if mode == 'nonrecursive' else values[:, 1]) * mask
    np.testing.assert_array_equal(column, expected)
    np.testing.assert_array_equal(output[:, 1], 7.)


@pytest.mark.parametrize('complement', [False, True])
def testMissingnessBasisTransformDispatch(complement):
    values, missing, root, parent, child, mask = makeCase(complement)
    basis = Basis(2)
    for term in [root, parent, child]:
        basis.append(term)
    output = np.full((4, 3), np.nan)
    basis.transform(values, missing, output)
    np.testing.assert_array_equal(
        output, np.column_stack([np.ones(4), values[:, 1], values[:, 1] * mask]),
    )


@pytest.mark.parametrize('complement', [False, True])
def testMissingnessPicklePreservesRecursiveApply(complement):
    values, missing, _, _, child, mask = makeCase(complement)
    restored = pickle.loads(pickle.dumps(child))
    output = np.zeros(4)
    restored.apply(values, missing, output)
    np.testing.assert_array_equal(output, values[:, 1] * mask)
