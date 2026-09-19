"""Replay the 180 original LBFGSB starts against explicit solver options."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize
from d01_optimizer_benchmark import makeData, makeModel
from UQPyL.surrogate.util.lbfgsb import LBFGSB

root = Path(__file__).parent
source = json.loads((root/'0918-d01-optimizer-results.json').read_text())['results']
rows = []
for row in source:
    if row['optimizer'] != 'LBFGSB':
        continue
    for old in row['optimizer_runs']:
        for label, options in [('default', {}), ('maxls50', {'maxls':50}), ('eps1e-6', {'eps':1e-6})]:
            model = makeModel(row['family'], row['profile'], 'LBFGSB', row['seed'])
            x, y = model.prepareTrainingData(*makeData(row['case'])[:2])
            infos, ub, lb = model.setting.getParaInfos(model.getParaList())
            if row['family'] == 'KRG':
                f, d = model._initialize(x)
            def objective(point):
                model.setting.setVals(infos, point)
                return model._objfunc(x,y) if row['family']=='GPR' else model._objFunc(y,f,d)
            captured = []
            def capture(*args, **kwargs):
                result = minimize(*args, **kwargs)
                captured.append(result)
                return result
            solver = LBFGSB(options)
            problem = SimpleNamespace(lb=lb, ub=ub, nInput=lb.size, objFunc=objective)
            initial = objective(np.array(old['first_point']))
            with patch('UQPyL.surrogate.util.lbfgsb.minimize', capture):
                point, score = solver.run(problem, xInit=old['first_point'])
            actual = float(objective(point))
            res = captured[0]
            rows.append(dict(family=row['family'], profile=row['profile'], case=row['case'], seed=row['seed'],
                             variant=label, success=bool(res.success), message=str(res.message), nfev=int(res.nfev),
                             score=float(score), actual=actual, initial=float(initial),
                             mismatch=bool(abs(actual-score)>1e-10+1e-8*max(abs(actual),abs(score))),
                             worse_than_start=bool(actual>initial+1e-10+1e-8*abs(initial))))
summary = {label:dict(runs=len(subset), failures=sum(not r['success'] for r in subset),
                      mismatches=sum(r['mismatch'] for r in subset), worse_than_start=sum(r['worse_than_start'] for r in subset),
                      evaluations=sum(r['nfev'] for r in subset))
           for label in ['default','maxls50','eps1e-6'] if (subset := [r for r in rows if r['variant']==label])}
import sys
Path(sys.argv[1]).write_text(json.dumps(dict(summary=summary, results=rows), indent=2)+'\n')
print(json.dumps(summary, indent=2))
