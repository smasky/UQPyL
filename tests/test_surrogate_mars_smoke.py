import numpy as np
import pytest

MARS = pytest.importorskip(
    "UQPyL.surrogate.mars.mars",
    reason="MARS extension modules are not available in this environment.",
).MARS


def test_mars_fit_predict_smoke():
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    model = MARS(max_terms=10, max_degree=1, penalty=2.0)
    model.fit(x, y)

    pred = model.predict(x[:5])
    assert pred.shape == (5, 1)
    assert np.isfinite(pred).all()


def test_mars_default_tune_parameters():
    model = MARS()

    assert list(model.getDefaultTuneParameters()) == ["max_terms", "max_degree", "penalty"]
    assert list(model.getDefaultTuneParameters(advanced=True)) == [
        "max_terms", "max_degree", "penalty", "endspan", "minspan", "thresh"
    ]
