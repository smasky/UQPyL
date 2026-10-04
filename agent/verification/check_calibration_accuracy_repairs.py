"""Independent post-fix accuracy controls; keep the original audit immutable."""
import json
from pathlib import Path
import warnings

import numpy as np

from UQPyL.calibration import IES, SUFI2
from UQPyL.problem import ModelProblem

root = Path(__file__).parent
baseline = json.loads((root/'1002-calibration-accuracy.json').read_text())
records = []
for case, forward, sigma in [('nonlinear_monotone',lambda x:x+.3*x**3,.3),('nonlinear_bimodal',lambda x:x**2,.2)]:
    reference = next(r for r in baseline if r['case']==case)
    problem=ModelProblem(nInput=1,lb=-10,ub=10,obs=np.array([[1.]]),simFunc=lambda x:forward(x[:,None,:]))
    for seed in range(10):
        prior=np.random.default_rng(seed).normal(size=(256,1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=IES(maxIters=20,seed=seed+100,adaptive=True,localLinearization=True).run(problem,prior,r=np.array([[sigma**2]]))
        values=result.posteriorDecs[:,0]
        old=next(r for r in baseline if r['case']==case and r['seed']==seed and r['n']==256 and r['method']=='IES_adaptive')
        records.append(dict(case=case,seed=seed,mean_error=float(abs(values.mean()-reference['exact_mean'])),
                            variance_relative_error=float(abs(values.var(ddof=1)-reference['exact_variance'])/reference['exact_variance']),
                            center_mass=float(np.mean(abs(values)<.5)),old_center_mass=old['center_mass'],
                            old_variance_relative_error=old['variance_relative_error'],
                            stop_reason=result.diagnostics['stopReason'],iterations=len(result.history.metricsHistory),
                            warnings=[str(w.message) for w in caught]))
for seed in range(100):
    rng=np.random.default_rng(10000+seed)
    truth=float(rng.normal())
    observation=truth+float(rng.normal(scale=.5))
    rng.normal(size=(128,1))  # Match the previous audit's GLUE prior pool exactly.
    prior=rng.normal(size=(4096,1))
    problem=ModelProblem(nInput=1,lb=-6,ub=6,obs=np.array([[observation],[2*observation]]),
                         simFunc=lambda x:np.stack([x[:,0],2*x[:,0]],axis=1)[:,:,None])
    result=SUFI2(nSamples=64,maxIters=4).run(problem,eliteSize=8,seed=seed,uncertaintyX=prior,
        logLikelihood=lambda obs,sim,mask:-.5*((sim[:,0]-obs[0])/.5)**2)
    uncertainty=result.extra['uncertainty']
    lower,upper=uncertainty['parameter_lower'][0],uncertainty['parameter_upper'][0]
    exactMean=observation/1.25
    records.append(dict(case='coverage',seed=seed,covered=bool(lower<=truth<=upper),width=float(upper-lower),
                        mean_error=float(abs(uncertainty['parameter_mean'][0]-exactMean)),
                        variance_error=float(abs(uncertainty['parameter_variance'][0]-.2)),
                        effective_sample_size=uncertainty['effective_sample_size']))
for seed in range(10):
    problem=ModelProblem(nInput=1,lb=0,ub=1,obs=np.array([[.731],[1.462]]),
                         simFunc=lambda x:np.stack([x[:,0],2*x[:,0]],axis=1)[:,:,None])
    result=SUFI2(nSamples=12,maxIters=5).run(problem,eliteSize=1,seed=seed)
    scores=[item['bestScore'] for item in result.history.metricsHistory]
    assert np.all(np.diff(scores)<=0)
    records.append(dict(case='incumbent',seed=seed,best_scores=scores))
(root/'1002-calibration-accuracy-repairs.json').write_text(json.dumps(records,indent=2)+'\n')
print('records',len(records))
for case in ['nonlinear_monotone','nonlinear_bimodal']:
    rows=[r for r in records if r['case']==case]
    print(case,{key:float(np.median([r[key] for r in rows])) for key in ['mean_error','variance_relative_error','center_mass','old_variance_relative_error','old_center_mass']},'warnings',sum(bool(r['warnings']) for r in rows))
rows=[r for r in records if r['case']=='coverage']
print('coverage',sum(r['covered'] for r in rows),'/100','median width',np.median([r['width'] for r in rows]),'mean error',np.median([r['mean_error'] for r in rows]),'variance error',np.median([r['variance_error'] for r in rows]))
