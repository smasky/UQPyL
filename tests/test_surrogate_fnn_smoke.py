import numpy as np

from UQPyL.surrogate.fnn.fully_connect_neural_network import FNN
from UQPyL.util.scaler import StandardScaler


def test_fnn_fit_predict_smoke_tiny_epoch():
    # Keep it extremely small so it runs fast and deterministically enough.
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    Y = (2 * X + 1).reshape(-1, 1)

    fnn = FNN(
        scalers=(StandardScaler(0, 1), StandardScaler(0, 1)),
        hidden_layer_sizes=[5],
        activation_functions="relu",
        solver="adam",
        learning_rate=0.01,
        epoch=3,
        batch_size=10,
        shuffle=False,
        no_improvement_count=10,
    )
    fnn.fit(X, Y)
    pred = fnn.predict(np.array([[0.25], [0.75]]))
    assert pred.shape == (2, 1)

