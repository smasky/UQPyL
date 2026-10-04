"""Accuracy audit against analytic moments, quadrature and known parameters.

Run in py312 with PYTHONPATH=. and one BLAS/OMP thread. No production edits.
"""
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.integrate import quad

from UQPyL.calibration import ES, IES, GLUE, SUFI2
from UQPyL.problem import ModelProblem

records = []


def runRecorded(method, problem, prior=None, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = method.run(problem, prior, **kwargs)
    return result, [str(item.message) for item in caught]


# Distinguish numerical agreement with sample-prior moments from Monte Carlo error.
for count in [32, 256]:
    for seed in range(10):
        prior = np.random.default_rng(seed).normal(size=(count, 2))
        matrix = np.array([[1., .5], [-.3, 1.]])
        obs, noise = np.array([.8, -.4]), np.array([[.25, .05], [.05, .4]])
        problem = ModelProblem(nInput=2, lb=-100, ub=100, obs=obs[:, None],
                               simFunc=lambda x:(x@matrix.T)[:, :, None])
        populationCov = np.linalg.inv(np.eye(2)+matrix.T@np.linalg.solve(noise, matrix))
        populationMean = populationCov@matrix.T@np.linalg.solve(noise, obs)
        sampleCov = np.cov(prior, rowvar=False)
        gain = np.linalg.solve(matrix@sampleCov@matrix.T+noise, matrix@sampleCov).T
        exactMean = prior.mean(0)+gain@(obs-matrix@prior.mean(0))
        exactCov = sampleCov-gain@matrix@sampleCov
        for name, method in [('ES', ES()), ('IES', IES(maxIters=5, seed=seed+100))]:
            result, caught = runRecorded(method, problem, prior, r=noise)
            posterior = result.posteriorDecs
            records.append(dict(case='linear_gaussian', method=name, n=count, seed=seed,
                                mean_error=float(np.linalg.norm(posterior.mean(0)-populationMean)),
                                covariance_error=float(np.linalg.norm(np.cov(posterior,rowvar=False)-populationCov)),
                                sample_mean_error=float(np.linalg.norm(posterior.mean(0)-exactMean)),
                                sample_covariance_error=float(np.linalg.norm(np.cov(posterior,rowvar=False)-exactCov)),warnings=caught))

# Exact nonlinear posteriors from independent 1D integration, not EnRML equations.
for case, forward, observation, sigma in [
    ('nonlinear_monotone', lambda x:x+.3*x**3, 1., .3),
    ('nonlinear_bimodal', lambda x:x**2, 1., .2),
]:
    def density(x):
        return np.exp(-.5*x*x-.5*((forward(x)-observation)/sigma)**2)
    def integrate(function):
        return sum(quad(function,a,b,epsabs=1e-12,epsrel=1e-11)[0] for a,b in [(-10,0),(0,10)])
    normalizer = integrate(density)
    exactMean = integrate(lambda x:x*density(x))/normalizer
    exactVariance = integrate(lambda x:(x-exactMean)**2*density(x))/normalizer
    centerMass = quad(density,-.5,.5,epsabs=1e-12)[0]/normalizer
    problem = ModelProblem(nInput=1,lb=-10,ub=10,obs=np.array([[observation]]),simFunc=lambda x:forward(x[:,None,:]))
    for seed in range(10):
        for count in [32,256]:
            prior = np.random.default_rng(seed).normal(size=(count,1))
            for name,method in [('ES',ES()),('IES_fixed',IES(maxIters=20,seed=seed+100)),
                                ('IES_adaptive',IES(maxIters=20,seed=seed+100,adaptive=True))]:
                result,caught = runRecorded(method,problem,prior,r=np.array([[sigma**2]]))
                values = result.posteriorDecs[:,0]
                records.append(dict(case=case,method=name,n=count,seed=seed,exact_mean=exactMean,
                                    exact_variance=exactVariance,mean_error=float(abs(values.mean()-exactMean)),
                                    variance_relative_error=float(abs(values.var(ddof=1)-exactVariance)/exactVariance),
                                    center_mass=float(np.mean(abs(values)<.5)),exact_center_mass=centerMass,
                                    best_output_error=float(abs(forward(result.bestDecs[0,0])-observation)),
                                    stop_reason=result.diagnostics.get('stopReason'),warnings=caught))
        for count in [256,4096]:
            prior = np.random.default_rng(seed).normal(size=(count,1))
            result,caught = runRecorded(GLUE(),problem,prior,threshold=np.inf,
                logLikelihood=lambda obs,sim,mask:-.5*np.sum(((sim-obs)/sigma)**2,axis=1))
            values,weights=prior[:,0],result.diagnostics['behavioralWeights']
            mean=float(weights@values)
            records.append(dict(case=case,method='GLUE_weighted',n=count,seed=seed,exact_mean=exactMean,
                                exact_variance=exactVariance,mean_error=abs(mean-exactMean),
                                variance_relative_error=float(abs(weights@(values-mean)**2-exactVariance)/exactVariance),
                                center_mass=float(weights@(abs(values)<.5)),exact_center_mass=centerMass,
                                effective_sample_size=result.diagnostics['effectiveSampleSize'],warnings=caught))

# Known noiseless optimum, with held-out prediction times.
for case in ['linear_fit','exponential_fit']:
    times=np.linspace(0,4,9)
    testTimes=np.linspace(.15,5,51)
    truth=np.array([2.,.7])
    def predict(x,t):
        return x[:,0,None]+x[:,1,None]*t if case=='linear_fit' else x[:,0,None]*np.exp(-x[:,1,None]*t)
    observation=predict(truth[None,:],times)
    problem=ModelProblem(nInput=2,lb=[0,.1],ub=[4,2],obs=observation.T,
                         simFunc=lambda x:predict(x,times)[:,:,None])
    for count in [32,128]:
        for seed in range(10):
            for name,options in [('SUFI2_guarded',{}),('SUFI2_envelope',dict(explorationFraction=0,minRangeFraction=0))]:
                result,caught=runRecorded(SUFI2(nSamples=count,maxIters=5,**options),problem,eliteSize=max(4,count//8),seed=seed)
                best=result.bestDecs
                records.append(dict(case=case,method=name,n=count,seed=seed,
                    normalized_parameter_error=float(np.linalg.norm((best[0]-truth)/np.array([4,1.9]))),
                    prediction_rmse=float(np.sqrt(np.mean((predict(best,testTimes)-predict(truth[None,:],testTimes))**2))),
                    elite_envelope_contains_truth=bool(np.all((truth>=result.diagnostics['updatedLb'])&(truth<=result.diagnostics['updatedUb']))),warnings=caught))

# Repeated prior-predictive data: nominal central 95% parameter intervals.
for seed in range(100):
    rng=np.random.default_rng(10000+seed)
    truth=float(rng.normal())
    observation=truth+float(rng.normal(scale=.5))
    exactMean,exactVariance=observation/1.25,.2
    records.append(dict(case='coverage',method='analytic',seed=seed,
                        covered=bool(abs(truth-exactMean)<=1.959963984540054*np.sqrt(exactVariance)),
                        width=2*1.959963984540054*np.sqrt(exactVariance)))
    problem=ModelProblem(nInput=1,lb=-100,ub=100,obs=np.array([[observation]]),simFunc=lambda x:x[:,None,:])
    prior=rng.normal(size=(128,1))
    for name,method in [('ES',ES()),('IES',IES(seed=seed+200,maxIters=3))]:
        result,caught=runRecorded(method,problem,prior,r=np.array([[.25]]))
        lower,upper=np.quantile(result.posteriorDecs[:,0],[.025,.975])
        records.append(dict(case='coverage',method=name,seed=seed,covered=bool(lower<=truth<=upper),width=float(upper-lower),warnings=caught))
    prior=rng.normal(size=(4096,1))
    result,caught=runRecorded(GLUE(),problem,prior,threshold=np.inf,
        logLikelihood=lambda obs,sim,mask:-.5*np.sum(((sim-obs)/.5)**2,axis=1))
    lower,upper=result.diagnostics['ppuLower'][0],result.diagnostics['ppuUpper'][0]
    records.append(dict(case='coverage',method='GLUE_weighted',seed=seed,covered=bool(lower<=truth<=upper),width=float(upper-lower),warnings=caught))
    # Both observations share one noise draw; SUFI2 receives no noise model.
    problem=ModelProblem(nInput=1,lb=-6,ub=6,obs=np.array([[observation],[2*observation]]),
                         simFunc=lambda x:np.stack([x[:,0],2*x[:,0]],axis=1)[:,:,None])
    result,caught=runRecorded(SUFI2(nSamples=64,maxIters=4),problem,eliteSize=8,seed=seed)
    lower,upper=np.quantile(result.eliteDecs[:,0],[.025,.975])
    records.append(dict(case='coverage',method='SUFI2_elite',seed=seed,covered=bool(lower<=truth<=upper),width=float(upper-lower),warnings=caught))

path=Path(__file__).with_name('1002-calibration-accuracy.json')
path.write_text(json.dumps(records,indent=2)+'\n')
print('records',len(records))
for case in sorted(set(r['case'] for r in records)):
    for method,count in sorted(set((r['method'],r.get('n',0)) for r in records if r['case']==case)):
        rows=[r for r in records if r['case']==case and r['method']==method and r.get('n',0)==count]
        summary={}
        for key in ['mean_error','covariance_error','sample_covariance_error','variance_relative_error','center_mass','best_output_error','normalized_parameter_error','prediction_rmse','width']:
            if key in rows[0]:
                summary[key]=[float(np.median([r[key] for r in rows])),float(np.max([r[key] for r in rows]))]
        for key in ['covered','elite_envelope_contains_truth']:
            if key in rows[0]:
                summary[key]=sum(r[key] for r in rows)
        summary['warning_runs']=sum(bool(r.get('warnings')) for r in rows)
        print(case,method,count,'runs',len(rows),json.dumps(summary))
print('nonlinear exact',[(r['case'],r['exact_mean'],r['exact_variance'],r['exact_center_mass']) for r in records if r['case'].startswith('nonlinear') and r['seed']==0 and r['n']==32 and r['method']=='ES'])
