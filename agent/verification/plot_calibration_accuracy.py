"""Illustrate one seed; main audit statistics use multiple seeds."""
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from UQPyL.calibration import ES, IES, GLUE
from UQPyL.problem import ModelProblem

root = Path(__file__).parent
rows = json.loads((root/'1002-calibration-accuracy.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
checks = []
for axis, case, forward, obs, sigma in [
    (axes[0], 'nonlinear_monotone', lambda x:x+.3*x**3, 1., .3),
    (axes[1], 'nonlinear_bimodal', lambda x:x**2, 1., .2),
]:
    grid = np.linspace(-10, 10, 200001)
    density = np.exp(-.5*grid**2-.5*((forward(grid)-obs)/sigma)**2)
    density /= np.trapezoid(density, grid)
    mean = np.trapezoid(grid*density, grid)
    variance = np.trapezoid((grid-mean)**2*density, grid)
    reference = next(r for r in rows if r['case']==case)
    assert abs(mean-reference['exact_mean'])<1e-10
    assert abs(variance-reference['exact_variance'])<1e-10
    checks.append(dict(case=case, grid_mean=float(mean), grid_variance=float(variance),
                       quadrature_mean=reference['exact_mean'], quadrature_variance=reference['exact_variance']))
    problem=ModelProblem(nInput=1,lb=-10,ub=10,obs=np.array([[obs]]),simFunc=lambda x:forward(x[:,None,:]))
    bins=np.linspace(-2,2,49)
    prior=np.random.default_rng(0).normal(size=(256,1))
    for name, method, color in [('ES',ES(),'#d97706'),('IES adaptive',IES(seed=100,maxIters=20,adaptive=True),'#dc2626')]:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            result=method.run(problem,prior,r=np.array([[sigma**2]]))
        axis.hist(result.posteriorDecs[:,0],bins=bins,weights=np.full(len(prior), 1 / len(prior) / (bins[1]-bins[0])),density=False,histtype='step',linewidth=1.5,label=name,color=color)
    prior=np.random.default_rng(0).normal(size=(4096,1))
    result=GLUE().run(problem,prior,threshold=np.inf,
                     logLikelihood=lambda obs,sim,mask:-.5*np.sum(((sim-obs)/sigma)**2,axis=1))
    axis.hist(prior[:,0],bins=bins,weights=result.diagnostics['behavioralWeights']/(bins[1]-bins[0]),density=False,
              histtype='step',linewidth=1.5,label='GLUE weighted',color='#2563eb')
    axis.plot(grid,density,color='#111827',label='Quadrature reference',linewidth=2)
    axis.set_xlim(-2,2)
    axis.set_xlabel('Parameter x')
    axis.set_ylabel('Posterior density')
    axis.set_title('Monotone: y = x + 0.3x³' if case.endswith('monotone') else 'Bimodal: y = x²')
    axis.legend(fontsize=8)
methods=['analytic','ES','IES','GLUE_weighted','SUFI2_elite']
coverage=[sum(r['covered'] for r in rows if r['case']=='coverage' and r['method']==m) for m in methods]
axes[2].bar(range(5),coverage,color=['#111827','#d97706','#dc2626','#2563eb','#6b7280'])
axes[2].set_xticks(range(5),['Analytic','ES','IES','GLUE','SUFI2\nelite'],rotation=20)
axes[2].axhline(95,ls='--',color='#6b7280',lw=1)
for i,value in enumerate(coverage):
    axes[2].text(i,value+1,str(value),ha='center',fontsize=10)
axes[2].set_ylim(0,108)
axes[2].set_ylabel('Truth inside interval / 100 trials')
axes[2].set_title('Linear noisy-data parameter coverage')
fig.suptitle('Calibration accuracy audit | Density plots: seed 0; coverage: 100 independent data sets',fontsize=12)
fig.savefig(root/'1002-calibration-accuracy.png',dpi=170)
(root/'1002-calibration-accuracy-quadrature-check.json').write_text(json.dumps(checks,indent=2)+'\n')
print(checks)
