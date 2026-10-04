"""Research prototype only: independent equations against pinned Equinor SIES.

Install reference outside production dependencies:
python -m pip install --no-deps --target .cache/calibration-reference/ies-0.2.7 iterative-ensemble-smoother==0.2.7
Run with py312, PYTHONPATH=. and single BLAS/OMP threads.
No third-party implementation is copied into the project.
"""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".cache/calibration-reference/ies-0.2.7"))
from iterative_ensemble_smoother import SIES


def squareRootEs(parameters, responses, observations, covariance):
    """Symmetric ensemble-space square-root update, derived from Kalman moments."""
    count = len(parameters)
    parameterAnomalies = parameters-parameters.mean(0)
    responseAnomalies = (responses-responses.mean(0)).T/np.sqrt(count-1)
    system = responseAnomalies@responseAnomalies.T+covariance
    solved = np.linalg.solve(system, responseAnomalies)
    gain = parameterAnomalies.T@solved.T/np.sqrt(count-1)
    mean = parameters.mean(0) + gain@(observations-responses.mean(0))
    transformCovariance = np.eye(count)-responseAnomalies.T@solved
    values, vectors = np.linalg.eigh((transformCovariance+transformCovariance.T)/2)
    assert values.min() > -1e-10
    transform = (vectors*np.sqrt(np.maximum(values,0)))@vectors.T
    return mean + transform@parameterAnomalies


def regressionEnrml(prior, current, responses, perturbedObservations, covariance, step):
    """Dense regression Gauss-Newton RML, anchored to the ORIGINAL ensemble.

    The prior covariance and perturbed targets stay fixed across iterations.
    This small-problem derivation is deliberately not a scalable production SIES.
    """
    initialCovariance = np.atleast_2d(np.cov(prior, rowvar=False))
    sensitivity = np.linalg.lstsq(current-current.mean(0), responses-responses.mean(0), rcond=None)[0].T
    system = sensitivity@initialCovariance@sensitivity.T+covariance
    gain = np.linalg.solve(system, sensitivity@initialCovariance).T
    innovation = perturbedObservations-responses+(current-prior)@sensitivity.T
    proposal = prior+innovation@gain.T
    return current+step*(proposal-current)


records = []
for seed in [11, 23, 47]:
    for dimension, count in [(1, 3), (2, 24), (8, 48), (20, 12)]:
        rng = np.random.default_rng(seed)
        prior = rng.normal(size=(count, dimension))
        matrix = rng.normal(size=(2, dimension))
        observations = np.array([0.6, -0.2])
        covariance = np.array([[0.5, 0.1], [0.1, 0.8]])
        priorCovariance = np.atleast_2d(np.cov(prior,rowvar=False))
        gain = np.linalg.solve(matrix@priorCovariance@matrix.T+covariance, matrix@priorCovariance).T
        exactMean = prior.mean(0)+gain@(observations-matrix@prior.mean(0))
        exactCovariance = priorCovariance-gain@matrix@priorCovariance
        corrected = squareRootEs(prior, prior@matrix.T, observations, covariance)
        meanError = float(np.max(np.abs(corrected.mean(0)-exactMean)))
        covarianceError = float(np.max(np.abs(np.atleast_2d(np.cov(corrected,rowvar=False))-exactCovariance)))
        assert meanError<1e-10 and covarianceError<1e-10
        reference = SIES(prior.T.copy(), covariance, observations, seed=seed, inversion="direct")
        targets = reference.D.T.copy()
        expectedMembers = prior+(targets-prior@matrix.T)@gain.T
        own, upstream = prior.copy(), prior.copy()
        for iteration in range(5):
            own = regressionEnrml(prior, own, own@matrix.T, targets, covariance, 1.)
            upstream = reference.sies_iteration((upstream@matrix.T).T, step_length=1.).T
            error = float(np.max(np.abs(own-upstream)))
            assert error<1e-9
            assert np.max(np.abs(own-expectedMembers))<1e-9
            records.append(dict(case="linear",seed=seed,dimension=dimension,n_ensemble=count,iteration=iteration+1,
                                reference_max_error=error,es_mean_error=meanError,es_covariance_error=covarianceError))

for seed in [11,23,47]:
    prior = np.random.default_rng(seed).normal(size=(40,1))
    observations = np.array([1.])
    covariance = np.array([[0.25]])
    reference = SIES(prior.T.copy(), covariance, observations, seed=seed, inversion="direct")
    targets = reference.D.T.copy()
    own, upstream = prior.copy(), prior.copy()
    def forward(x):
        return x+0.3*x**3
    for iteration in range(8):
        own = regressionEnrml(prior, own, forward(own), targets, covariance, .5)
        upstream = reference.sies_iteration(forward(upstream).T, step_length=.5).T
        error = float(np.max(np.abs(own-upstream)))
        assert error<1e-8
        records.append(dict(case="nonlinear_cubic",seed=seed,iteration=iteration+1,reference_max_error=error,
                            mean=float(own.mean()),variance=float(own.var(ddof=1))))

simple = np.array([[-1.],[0.],[1.]])
corrected = squareRootEs(simple,simple,np.array([1.]),np.array([[1.]]))
records.append(dict(case="original_scalar_counterexample",corrected_mean=float(corrected.mean()),
                    corrected_variance=float(corrected.var(ddof=1)),expected_mean=.5,expected_variance=.5))
output = Path(__file__).with_name("1002-es-ies-reference-prototype.json")
output.write_text(json.dumps(records,indent=2)+"\n")
print("records",len(records))
print("max upstream difference",max(r.get("reference_max_error",0) for r in records))
print("max ES covariance error",max(r.get("es_covariance_error",0) for r in records))
print(records[-1])
