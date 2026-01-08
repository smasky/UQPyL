import numpy as np

from UQPyL.surrogate.util.adam import Adam
from UQPyL.problem.sop.single_simple_problem import Sphere


def test_adam_update_and_run_smoke_on_sphere_objective():
    # Minimize sphere(x) with a single parameter vector using numeric gradient.
    problem = Sphere(nInput=2, ub=1.0, lb=-1.0)

    x = np.array([0.5, -0.25], dtype=float)
    params = [x]

    def loss_and_grad(p):
        x = p[0]
        fx = float(problem.objFunc(x.reshape(1, -1))[0, 0])
        eps = 1e-6
        g = np.zeros_like(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            fp = float(problem.objFunc(xp.reshape(1, -1))[0, 0])
            g[i] = (fp - fx) / eps
        return fx, [g]

    opt = Adam(params=params, learning_rate=0.05, epoch=5)
    _, best = opt.run(params, loss_and_grad, arg=(params,))
    assert np.isfinite(best)
    assert len(opt.loss_curve) == 5

