"""Compare production IES to isolated Equinor SIES 0.2.7, with identical targets.

Run with PYTHONPATH=. in py312; reference install instructions are in
prototype_es_ies_reference.py. No reference source is copied or shipped.
"""
import json
from pathlib import Path
import sys

import numpy as np

from UQPyL.calibration import IES
from UQPyL.problem import ModelProblem

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.cache/calibration-reference/ies-0.2.7'))
from iterative_ensemble_smoother import SIES

records = []
for seed in [11, 23, 47]:
    for dimension, count in [(1, 12), (2, 24), (8, 48), (20, 12)]:
        for nonlinear in [False, True]:
            rng = np.random.default_rng(seed)
            prior = rng.normal(size=(count, dimension))
            matrix = rng.normal(size=(2, dimension)) / np.sqrt(dimension)
            obs = np.array([.6, -.2])
            noise = np.array([[.5, .1], [.1, .8]])
            def forward(x):
                transformed = x + .05 * x**3 if nonlinear else x
                return transformed @ matrix.T
            reference = SIES(prior.T.copy(), noise, obs, seed=seed, inversion='direct')
            targets = reference.D.T.copy()
            class MatchedTargetsIes(IES):
                def _prepareIteration(self, X, obs, r):
                    context = super()._prepareIteration(X, obs, r)
                    context['targets'] = targets.copy()
                    return context
            problem = ModelProblem(nInput=dimension, lb=-1e8, ub=1e8, obs=obs[:, None],
                                   simFunc=lambda x: forward(x)[:, :, None])
            upstream = prior.copy()
            for iteration in range(1, 6):
                upstream = reference.sies_iteration(forward(upstream).T, step_length=1.).T
                production = MatchedTargetsIes(maxIters=iteration, seed=seed).run(problem, prior, r=noise)
                error = float(np.max(np.abs(production.posteriorDecs - upstream)))
                records.append(dict(seed=seed, dimension=dimension, n_ensemble=count,
                                    nonlinear=nonlinear, iteration=iteration, reference_max_error=error))
                assert error < 2e-9, records[-1]
output = Path(__file__).with_name('1002-es-ies-production-reference.json')
output.write_text(json.dumps(records, indent=2) + '\n')
print('records', len(records))
print('max difference', max(r['reference_max_error'] for r in records))
