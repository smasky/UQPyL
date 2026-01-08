import numpy as np
import pytest

from UQPyL.optimization.population import Population


def test_population_add_overloads_and_getitem_type_error():
    p = Population(np.array([[0.0, 0.0]]), np.array([[1.0]]))
    q = Population(np.array([[1.0, 1.0]]), np.array([[2.0]]))

    p.add(q)
    assert len(p) == 2

    p.add(np.array([[2.0, 2.0]]), np.array([[3.0]]))
    assert len(p) == 3

    with pytest.raises(TypeError):
        _ = p["bad"]

