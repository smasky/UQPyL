import numpy as np

from UQPyL.problem.sop.single_simple_problem import Sphere
from UQPyL.surrogate.gp.gaussian_process import GPR
from UQPyL.surrogate.util.boxmin import Boxmin
from UQPyL.util.scaler import StandardScaler


def test_boxmin_covered_via_gpr_fit_with_real_optimizer():
    # Use Sphere to generate a clean 1D regression dataset; keep it tiny for speed.
    problem = Sphere(nInput=1, ub=1.0, lb=-1.0)
    X = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    Y = problem.objFunc(X)

    gpr = GPR(
        scalers=(StandardScaler(0, 1), StandardScaler(0, 1)),
        optimizer=Boxmin(),
        nRestartTimes=0,
    )
    gpr.fit(X, Y)
    pred = gpr.predict(np.array([[0.0], [0.5]]))
    assert pred.shape == (2, 1)

