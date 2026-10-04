"""Public-API experiments for guarded sampling and nonlinear RML backtracking."""
import json
from pathlib import Path
import warnings

import numpy as np

from UQPyL.calibration import IES, SUFI2
from UQPyL.problem import ModelProblem

records = []
for seed in range(10):
    problem = ModelProblem(nInput=1, lb=0, ub=1, obs=np.array([[.731], [1.462]]),
                           simFunc=lambda x: np.stack([x[:, 0], 2*x[:, 0]], axis=1)[:, :, None])
    errors = []
    for guards in [False, True]:
        options = {} if guards else dict(explorationFraction=0, minRangeFraction=0)
        result = SUFI2(nSamples=12, maxIters=5, **options).run(problem, eliteSize=1, seed=seed)
        errors.append(float(result.diagnostics['scores'].min()))
    records.append(dict(case='sufi_single_elite', seed=seed, old_error=errors[0], guarded_error=errors[1]))

for name, function, limits, observation, variance in [
    ('cubic', lambda x:x**3, (.05, .15), 1., 1e-5),
    ('exponential', np.exp, (-2.1, -1.9), 2., .01),
    ('sine', np.sin, (1.4, 1.7), -.5, .01),
]:
    prior = np.linspace(*limits, 30)[:, None]
    problem = ModelProblem(nInput=1, lb=-6, ub=6, obs=np.array([[observation]]),
                           simFunc=lambda x:function(x[:, None, :]))
    for seed in [11, 23, 47]:
        targets = observation + np.random.default_rng(seed).standard_normal(prior.shape)*np.sqrt(variance)
        def cost(x):
            return float(np.mean((x-prior)**2/prior.var(ddof=1)+(function(x)-targets)**2/variance))
        for adaptive in [False, True]:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                result = IES(maxIters=20, seed=seed, adaptive=adaptive).run(problem, prior, r=np.array([[variance]]))
            before, after = cost(prior), cost(result.posteriorDecs)
            if adaptive:
                assert after <= before + 1e-9
                for trial in result.diagnostics['lineSearch']:
                    if trial['accepted']:
                        assert trial['merit_after'][1] <= trial['merit_before'][1] + 1e-8
            records.append(dict(case=name, seed=seed, adaptive=adaptive, initial_cost=before, final_cost=after,
                                stop_reason=result.diagnostics['stopReason'], iterations=len(result.history.metricsHistory),
                                trial_count=len(result.diagnostics.get('lineSearch', [])),
                                warnings=[str(w.message) for w in captured]))
output = Path(__file__).with_name('1002-calibration-followup.json')
output.write_text(json.dumps(records, indent=2)+'\n')
print('records', len(records))
sufi = [r for r in records if r['case']=='sufi_single_elite']
print('SUFI improvements', sum(r['guarded_error']<r['old_error'] for r in sufi), '/',len(sufi))
for name in ['cubic','exponential','sine']:
    rows = [r for r in records if r['case']==name]
    print(name, [(r['seed'],r['adaptive'],r['final_cost'],r['stop_reason']) for r in rows])
