"""Independent calibration references and public workflow probes, py312."""

import json
from pathlib import Path
import warnings

import numpy as np

from UQPyL.calibration import ES, IES, GLUE, SUFI2
from UQPyL.calibration.util import rmse
from UQPyL.problem import ModelProblem


records = []


def record(case, **data):
    records.append(dict(case=case, **data))


def problemFor(matrix, obs, **kwargs):
    return ModelProblem(
        nInput=matrix.shape[1],
        lb=-100,
        ub=100,
        obs=np.asarray(obs)[:, None],
        simFunc=lambda x: (np.asarray(x) @ matrix.T)[:, :, None],
        **kwargs,
    )


# Independent dense covariance equations, including correlated observation errors.
for seed in [11, 23, 47]:
    x = np.random.default_rng(seed).normal(size=(24, 2))
    matrix = np.array([[1.0, 0.5], [-0.3, 2.0]])
    obs = np.array([0.7, -0.2])
    for noise in [0, 0.1, 1]:
        covariance = noise * np.array([[1.0, 0.3], [0.3, 2.0]])
        for iterations in [1, 3]:
            for regularization in [0, 0.2]:
                expected = x.copy()
                for _ in range(iterations):
                    predictions = expected @ matrix.T
                    dx = expected - expected.mean(0)
                    dy = predictions - predictions.mean(0)
                    gain = (dx.T @ dy / 23) @ np.linalg.pinv(dy.T @ dy / 23 + covariance + regularization * np.eye(2))
                    expected += (obs - predictions) @ gain.T
                model = IES(maxIters=iterations, lam=regularization)
                result = model.run(problemFor(matrix, obs), x, r=covariance)
                difference = float(np.max(np.abs(expected - result.posteriorDecs)))
                record(
                    "ies_dense_reference",
                    seed=seed,
                    noise=noise,
                    iterations=iterations,
                    regularization=regularization,
                    max_error=difference,
                    correct=difference < 1e-8,
                )
                if iterations == 1 and regularization == 0:
                    es = ES().run(problemFor(matrix, obs), x, r=covariance)
                    record(
                        "es_dense_reference",
                        seed=seed,
                        noise=noise,
                        max_error=float(np.max(np.abs(es.posteriorDecs - expected))),
                    )

# A deterministic common-observation update is not a posterior covariance sampler.
x = np.array([[-1.0], [0.0], [1.0]])  # sample mean 0, sample variance 1
p = problemFor(np.array([[1.0]]), [1.0])
for steps in [1, 3, 10]:
    result = IES(maxIters=steps).run(p, x, r=np.array([[1.0]]))
    record(
        "posterior_interpretation",
        iterations=steps,
        actual_mean=float(result.posteriorDecs.mean()),
        actual_variance=float(result.posteriorDecs.var(ddof=1)),
        gaussian_posterior_mean=0.5,
        gaussian_posterior_variance=0.5,
    )

# Hand-checked screening and empirical quantiles, without model uncertainty claims.
matrix = np.eye(2)
obs = np.array([1.0, 2.0])
x = np.array([[1.0, 2.0], [1.2, 2.2], [0.0, 0.0], [0.7, 1.7]])
scores = np.sqrt(np.mean((x - obs) ** 2, axis=1))
p = problemFor(matrix, obs)
glue = GLUE().run(p, x, threshold=0.25)
record("glue_screening", correct=bool(np.array_equal(glue.diagnostics["behavioralMask"], scores <= 0.25)))
sufi = SUFI2().run(p, x, eliteSize=3)
elite = x[np.argsort(scores)[:3]]
record(
    "sufi_quantile",
    correct=bool(
        np.allclose(sufi.diagnostics["ppuLower"], np.quantile(elite, 0.025, axis=0))
        and np.allclose(sufi.diagnostics["ppuUpper"], np.quantile(elite, 0.975, axis=0))
        and np.allclose(sufi.diagnostics["updatedLb"], elite.min(0))
        and np.allclose(sufi.diagnostics["updatedUb"], elite.max(0))
    ),
)

for variableType in [1, 2]:
    seen = []

    def simulate(values):
        seen.append(np.asarray(values).copy())
        return np.stack([values[:, 0], 2 * values[:, 0]], axis=1)[:, :, None]

    kwargs = {"varSet": {0: [0.25, 0.75]}} if variableType == 2 else {}
    p = ModelProblem(
        nInput=1, lb=0, ub=2, varType=[variableType], obs=np.array([[1.0], [2.0]]), simFunc=simulate, **kwargs
    )
    result = SUFI2(maxIters=2, nSamples=12).run(p, eliteSize=4, seed=11)
    values = np.concatenate(seen).ravel()
    valid = np.isin(values, [0.25, 0.75]) if variableType == 2 else values == np.round(values)
    record(
        "sufi_internal_domain",
        variable_type=variableType,
        invalid_count=int(np.count_nonzero(~valid)),
        example=values[:5].tolist(),
    )
    supplied = np.array([[0.25], [0.75]]) if variableType == 2 else np.array([[0.0], [1.0], [2.0]])
    seen.clear()
    SUFI2().run(p, supplied, eliteSize=2)
    record("sufi_supplied_domain", variable_type=variableType, correct=bool(np.array_equal(seen[0], supplied)))

for scale in [1.0, 1e-200, 1e200]:
    obs = np.array([1.0, 2.0]) * scale
    simulations = np.array([[1.1, 2.2], [1.0, 2.1]]) * scale
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = rmse(obs, simulations)
    expected = np.sqrt(np.mean((simulations / scale - obs / scale) ** 2, axis=1)) * scale
    record(
        "rmse_range",
        scale=scale,
        actual=[str(v) for v in actual],
        expected=expected.tolist(),
        warnings=[str(item.message) for item in caught],
    )
    p = ModelProblem(
        nInput=1, lb=0, ub=1, obs=obs[:, None], simFunc=lambda x: simulations[np.asarray(x[:, 0], dtype=int), :, None]
    )
    candidates = np.array([[0.0], [1.0]])
    for name in ["GLUE", "SUFI2"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = (
                    GLUE().run(p, candidates, threshold=0.1 * scale)
                    if name == "GLUE"
                    else SUFI2().run(p, candidates, eliteSize=1)
                )
                record(
                    "rmse_workflow",
                    scale=scale,
                    method=name,
                    best=float(result.bestDecs.item()),
                    expected_best=1.0,
                    behavioral_mask=result.diagnostics.get("behavioralMask", np.array([])).tolist(),
                )
            except Exception as error:
                record("rmse_workflow", scale=scale, method=name, error=f"{type(error).__name__}: {error}")

Path(__file__).with_name("1002-calibration-science-after-rmse.json").write_text(json.dumps(records, indent=2) + "\n")
for row in records:
    print(row)
